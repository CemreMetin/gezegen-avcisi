import os
import gc
import datetime
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from pipeline.vetos import run_analytical_vetos
from pipeline.preprocessing import get_binned_views, fetch_spoc_urls, download_lightcurve_from_uri

try:
    from altaipony.flarelc import FlareLightCurve
    HAS_ALTAIPONY = True
except Exception:
    HAS_ALTAIPONY = False

DATA_DIR = "data"
PENDING_DIR = "pending_candidates"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "300"))
MAX_PERIOD_DAYS = 11.0  # Tek sektor (~27 gun) icinde en az 2-3 gecis gormek icin


def _safe_float(val, default):
    try:
        if val is None:
            return default
        f = float(val)
        return f if np.isfinite(f) and f > 0 else default
    except Exception:
        return default


def _detect_and_mask_flares(lc, tic_str, flares_csv):
    """AltaiPony ile yildiz patlamalarini tespit edip maskeler; hata olursa ust-uc deger kirpar."""
    clean_mask = np.ones(len(lc), dtype=bool)
    if not HAS_ALTAIPONY:
        return lc.remove_outliers(sigma_upper=3.5, sigma_lower=10.0)

    try:
        flux_err = lc.flux_err if getattr(lc, "flux_err", None) is not None else None
        flc = FlareLightCurve(time=lc.time, flux=lc.flux, flux_err=flux_err)
        detrended = flc.detrend("savgol")
        if isinstance(detrended, tuple):
            detrended = detrended[0]
        detrended = detrended.find_flares(N1=3, N2=3, N3=3)

        if hasattr(detrended, "flares") and detrended.flares is not None and not detrended.flares.empty:
            flares_df = detrended.flares.copy()
            for _, row in flares_df.iterrows():
                tstart = row.get("tstart", None)
                tstop = row.get("tstop", None)
                if tstart is not None and tstop is not None:
                    flare_idx = (lc.time.value >= float(tstart) - 0.01) & (lc.time.value <= float(tstop) + 0.01)
                    clean_mask[flare_idx] = False
            flares_df.insert(0, "tic_id", tic_str)
            write_header = not os.path.exists(flares_csv) or os.path.getsize(flares_csv) == 0
            flares_df.to_csv(flares_csv, mode="a", header=write_header, index=False)
            return lc[clean_mask]
    except Exception as e:
        print(f"AltaiPony uyarisi TIC {tic_str}: {e} (ust-uc kirpmaya geciliyor)")
        return lc.remove_outliers(sigma_upper=3.5, sigma_lower=10.0)

    return lc[clean_mask]


def process_batch(batch_size=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PENDING_DIR, exist_ok=True)

    limit = batch_size if batch_size is not None else BATCH_SIZE
    cursor_file = os.path.join(DATA_DIR, "cursor.txt")
    queue_file = os.path.join(DATA_DIR, "tic_queue.txt")
    flares_csv = os.path.join(DATA_DIR, "flares.csv")
    eb_csv = os.path.join(DATA_DIR, "eclipsing_binaries.csv")

    start_idx = int(open(cursor_file).read().strip()) if os.path.exists(cursor_file) else 0
    if not os.path.exists(queue_file):
        print("Kuyruk dosyasi bulunamadi: data/tic_queue.txt")
        return

    all_tics = [line.strip() for line in open(queue_file) if line.strip()]
    batch_tics = all_tics[start_idx : start_idx + limit]

    if not batch_tics:
        print("Kuyruk tamamlandi veya bos.")
        return

    print(f"MAST'tan {len(batch_tics)} hedef icin SPOC FITS adresleri toplu cekiliyor...")
    url_map = fetch_spoc_urls(batch_tics, chunk_size=100)
    print(f"Indirilecek SPOC hedef sayisi: {len(url_map)}")

    collected_tics, collected_g, collected_l, collected_scalars, collected_sde = [], [], [], [], []
    processed_count = 0

    for tic_str in batch_tics:
        data_uri = url_map.get(tic_str)
        if not data_uri:
            continue

        try:
            lc = download_lightcurve_from_uri(data_uri, timeout=25)
            if lc is None:
                continue

            meta = getattr(lc, "meta", {}) or {}
            r_star = _safe_float(meta.get("RADIUS"), 1.0)
            teff = _safe_float(meta.get("TEFF"), 5500.0)

            lc = lc.remove_nans().normalize()
            if len(lc) < 200:
                continue

            lc = _detect_and_mask_flares(lc, tic_str, flares_csv)
            if len(lc) < 200:
                continue

            # 2 dk kadansta 501 nokta (~16.7 saat) gezegen gecisini torpulemeden yildiz egilimini temizler
            win = min(501, len(lc) if len(lc) % 2 == 1 else len(lc) - 1)
            flat_lc = lc.flatten(window_length=max(win, 101)).remove_outliers(sigma_lower=10.0, sigma_upper=5.0)
            t = np.asarray(flat_lc.time.value, dtype=np.float64)
            f = np.asarray(flat_lc.flux.value, dtype=np.float64)

            valid = np.isfinite(t) & np.isfinite(f)
            t, f = t[valid], f[valid]
            if len(t) < 200:
                continue

            time_span = t.max() - t.min()
            max_p = min(MAX_PERIOD_DAYS, time_span / 2.2)
            if max_p <= 0.6:
                continue

            bls = BoxLeastSquares(t, f)
            period_grid = np.linspace(0.5, max_p, 4000)
            duration_grid = np.linspace(0.04, 0.22, 10)
            pg = bls.power(period_grid, duration_grid)

            power = np.asarray(pg.power, dtype=np.float64)
            std_p = float(np.std(power))
            if std_p <= 0:
                continue

            best_idx = int(np.argmax(power))
            sde = float((power[best_idx] - np.mean(power)) / std_p)
            processed_count += 1

            if sde >= 7.0:
                p_best = float(pg.period[best_idx])
                t0_best = float(pg.transit_time[best_idx])
                dur_best = float(pg.duration[best_idx])

                passed, reason, metrics = run_analytical_vetos(t, f, p_best, t0_best, dur_best, r_star)
                if passed:
                    g_view, l_view = get_binned_views(t, f, p_best, t0_best, dur_best)
                    dur_hours = dur_best * 24.0
                    collected_tics.append(tic_str)
                    collected_g.append(g_view)
                    collected_l.append(l_view)
                    collected_scalars.append([p_best, metrics["depth"], metrics["rp"], r_star, teff, dur_hours])
                    collected_sde.append(sde)
                    print(f"[ADAY] TIC {tic_str} | P={p_best:.3f}g | SDE={sde:.1f} | Rp={metrics['rp']:.2f} R_Jup")
                elif metrics.get("type") == "EB":
                    write_header = not os.path.exists(eb_csv) or os.path.getsize(eb_csv) == 0
                    pd.DataFrame([{
                        "tic_id": tic_str,
                        "period": round(p_best, 5),
                        "depth": round(metrics.get("depth", 0.0), 6),
                        "rp_jup": round(metrics.get("rp", 0.0), 3),
                        "reason": reason,
                    }]).to_csv(eb_csv, mode="a", header=write_header, index=False)

        except Exception as e:
            print(f"Hata TIC {tic_str}: {e}")
        finally:
            gc.collect()

    if collected_tics:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M")
        out_npz = os.path.join(PENDING_DIR, f"batch_{stamp}.npz")
        np.savez_compressed(
            out_npz,
            tic_ids=np.array(collected_tics),
            global_views=np.array(collected_g, dtype=np.float32),
            local_views=np.array(collected_l, dtype=np.float32),
            scalars=np.array(collected_scalars, dtype=np.float32),
            sde=np.array(collected_sde, dtype=np.float32),
        )
        print(f"Kaydedildi: {out_npz} ({len(collected_tics)} aday)")
    else:
        print(f"Bu turda ({processed_count}/{len(batch_tics)} islenen yildiz) yeni aday bulunamadi.")

    with open(cursor_file, "w") as f_cur:
        f_cur.write(str(start_idx + len(batch_tics)))


if __name__ == "__main__":
    process_batch()
