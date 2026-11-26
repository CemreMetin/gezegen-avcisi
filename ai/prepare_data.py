import os
import gc
import argparse
import requests
import numpy as np
import pandas as pd
from pipeline.preprocessing import get_binned_views, fetch_spoc_urls, download_lightcurve_from_uri

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
OUTPUT_NPZ = "data/tess_tce_training.npz"
TESS_BTJD_OFFSET = 2457000.0


def _save_dataset(path, global_list, local_list, scalar_list, labels_list, tic_list):
    np.savez_compressed(
        path,
        global_views=np.array(global_list, dtype=np.float32),
        local_views=np.array(local_list, dtype=np.float32),
        scalars=np.array(scalar_list, dtype=np.float32),
        labels=np.array(labels_list, dtype=np.float32),
        tic_ids=np.array(tic_list, dtype=np.int64),
    )


def build_training_dataset(limit_per_class=150):
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)

    # NASA Exoplanet Archive resmi TESS Objects of Interest (toi) tablosu
    query = f"""
    SELECT TOP {limit_per_class * 8}
        tid, pl_orbper, pl_tranmid, pl_trandurh, pl_trandep,
        pl_rade, st_teff, st_rad, st_tmag, tfopwg_disp
    FROM toi
    WHERE pl_orbper IS NOT NULL
      AND pl_orbper BETWEEN 0.4 AND 25.0
      AND pl_tranmid IS NOT NULL
      AND pl_trandurh IS NOT NULL
      AND pl_trandurh > 0
      AND pl_trandep IS NOT NULL
      AND pl_trandep > 0
      AND tfopwg_disp IN ('CP', 'KP', 'PC', 'FP', 'FA')
    ORDER BY st_tmag ASC
    """
    print("NASA Exoplanet Archive (TOI) sorgulaniyor...")
    resp = requests.get(TAP_URL, params={"query": query, "format": "json"}, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())

    if df.empty:
        raise RuntimeError("NASA TAP API bos sonuc dondurdu.")

    df["label"] = df["tfopwg_disp"].isin(["CP", "KP", "PC"]).astype(float)
    pos_df = df[df["label"] == 1.0].drop_duplicates(subset=["tid"]).head(limit_per_class * 2)
    neg_df = df[df["label"] == 0.0].drop_duplicates(subset=["tid"]).head(limit_per_class * 2)
    table = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    candidate_tics = [str(int(t)) for t in table["tid"].tolist()]
    print(f"MAST'tan {len(candidate_tics)} aday icin SPOC FITS adresleri toplu cekiliyor...")
    url_map = fetch_spoc_urls(candidate_tics, chunk_size=100)
    print(f"SPOC verisi bulunan aday sayisi: {len(url_map)}")

    global_list, local_list, scalar_list, labels_list, tic_list = [], [], [], [], []
    counts = {1.0: 0, 0.0: 0}

    for _, row in table.iterrows():
        lbl = float(row["label"])
        if counts[lbl] >= limit_per_class:
            if counts[1.0] >= limit_per_class and counts[0.0] >= limit_per_class:
                break
            continue

        tic_id = int(row["tid"])
        data_uri = url_map.get(str(tic_id))
        if not data_uri:
            continue

        try:
            lc = download_lightcurve_from_uri(data_uri, timeout=25)
            if lc is None:
                continue

            lc_clean = lc.remove_nans().normalize()
            if len(lc_clean) < 200:
                continue
            win = min(501, len(lc_clean) if len(lc_clean) % 2 == 1 else len(lc_clean) - 1)
            flat_lc = lc_clean.flatten(window_length=max(win, 101)).remove_outliers(sigma_lower=10.0, sigma_upper=5.0)
            t = np.asarray(flat_lc.time.value, dtype=np.float64)
            f = np.asarray(flat_lc.flux.value, dtype=np.float64)
            valid = np.isfinite(t) & np.isfinite(f)
            t, f = t[valid], f[valid]
            if len(t) < 200:
                continue

            period = float(row["pl_orbper"])
            t0_raw = float(row["pl_tranmid"])
            t0_btjd = t0_raw - TESS_BTJD_OFFSET if t0_raw > 2400000.0 else t0_raw
            dur_hours = float(row["pl_trandurh"])
            dur_days = dur_hours / 24.0
            depth_frac = float(row["pl_trandep"]) / 1e6

            r_star = float(row["st_rad"]) if pd.notnull(row["st_rad"]) and float(row["st_rad"]) > 0 else 1.0
            teff = float(row["st_teff"]) if pd.notnull(row["st_teff"]) and float(row["st_teff"]) > 0 else 5500.0
            if pd.notnull(row["pl_rade"]) and float(row["pl_rade"]) > 0:
                rp_jup = float(row["pl_rade"]) * 0.08921
            else:
                rp_jup = (r_star * np.sqrt(max(depth_frac, 0.0))) / 0.10045

            g_view, l_view = get_binned_views(t, f, period, t0_btjd, dur_days)
            scalars = [period, depth_frac, rp_jup, r_star, teff, dur_hours]

            global_list.append(g_view)
            local_list.append(l_view)
            scalar_list.append(scalars)
            labels_list.append(lbl)
            tic_list.append(tic_id)
            counts[lbl] += 1

            tag = "PC" if lbl == 1.0 else "FP"
            print(f"[{counts[1.0]}/{limit_per_class} PC | {counts[0.0]}/{limit_per_class} FP] TIC {tic_id} ({row['tfopwg_disp']} -> {tag})")

            if len(tic_list) % 25 == 0:
                _save_dataset(OUTPUT_NPZ, global_list, local_list, scalar_list, labels_list, tic_list)
        except Exception as e:
            print(f"Atlandi TIC {tic_id}: {e}")
        finally:
            gc.collect()

    _save_dataset(OUTPUT_NPZ, global_list, local_list, scalar_list, labels_list, tic_list)
    print(f"Egitim paketi hazir: {OUTPUT_NPZ} (Toplam: {len(tic_list)} | PC: {counts[1.0]}, FP: {counts[0.0]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=150, help="Sinif basina ornek sayisi (varsayilan: 150)")
    args = parser.parse_args()
    build_training_dataset(limit_per_class=args.limit)
