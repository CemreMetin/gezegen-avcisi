import os
import tempfile
import requests
import numpy as np
import lightkurve as lk
from astroquery.mast import Observations

Observations.TIMEOUT = 30
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"


def fetch_spoc_urls(tic_ids, chunk_size=100):
    """TIC listesi icin MAST'tan toplu halde SPOC *_lc.fits URL'lerini ceker (cok hizli, name-lookup yok)."""
    url_map = {}
    tic_strs = [str(t).strip() for t in tic_ids]

    for i in range(0, len(tic_strs), chunk_size):
        chunk = tic_strs[i : i + chunk_size]
        try:
            obs = Observations.query_criteria(
                obs_collection="TESS",
                provenance_name="SPOC",
                dataproduct_type="timeseries",
                target_name=chunk,
            )
            if len(obs) == 0:
                continue
            for row in obs:
                tic = str(row["target_name"]).strip().lstrip("0")
                data_url = str(row["dataURL"]) if row["dataURL"] else ""
                if tic and tic not in url_map and data_url.endswith("_lc.fits"):
                    url_map[tic] = data_url
        except Exception as e:
            print(f"MAST toplu sorgu uyarisi (parca {i}): {e}")

    return url_map


def download_lightcurve_from_uri(data_uri, timeout=25):
    """Tek bir MAST FITS dosyasini kisa timeout ile indirip LightCurve olarak okur ve gecici dosyayi siler."""
    resp = requests.get(MAST_DOWNLOAD_URL, params={"uri": data_uri}, timeout=timeout)
    resp.raise_for_status()

    fd, tmp_path = tempfile.mkstemp(suffix=".fits")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(resp.content)
        lc = lk.read(tmp_path)
        return lc
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def get_binned_views(time, flux, period, t0, duration_days):
    """Faz katlamali global (201) ve lokal (61) gorunumleri AstroNet standardinda uretir.

    Cikti dizileri medyan=0, gecis dibi~=-1 olacak sekilde normalize edilir.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    valid = np.isfinite(time) & np.isfinite(flux)
    time = time[valid]
    flux = flux[valid]

    if len(time) < 10 or period <= 0:
        return np.zeros(201, dtype=np.float32), np.zeros(61, dtype=np.float32)

    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    sort_idx = np.argsort(phase)
    p_sorted = phase[sort_idx]
    f_sorted = flux[sort_idx]

    g_view = _bin_series(p_sorted, f_sorted, -0.5, 0.5, 201)

    half_phase_window = min(0.5, max((2.0 * duration_days) / period, 0.02))
    local_mask = np.abs(p_sorted) <= half_phase_window
    p_loc = p_sorted[local_mask]
    f_loc = f_sorted[local_mask]

    if len(p_loc) >= 3:
        l_view = _bin_series(p_loc, f_loc, -half_phase_window, half_phase_window, 61)
    else:
        l_view = np.full(61, np.median(f_sorted), dtype=np.float64)

    g_view = g_view - np.median(g_view)
    l_view = l_view - np.median(l_view)

    # AstroNet standardi: gecis dibini ~ -1.0 olcegine getir (ham derinlik skalerde korunur)
    scale = max(abs(float(np.min(l_view))), abs(float(np.min(g_view))), 1e-6)
    g_view = np.clip(g_view / scale, -2.0, 2.0)
    l_view = np.clip(l_view / scale, -2.0, 2.0)

    return g_view.astype(np.float32), l_view.astype(np.float32)


def _bin_series(x, y, xmin, xmax, n_bins):
    edges = np.linspace(xmin, xmax, n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

    binned = np.full(n_bins, np.nan, dtype=np.float64)
    for i in range(n_bins):
        mask = idx == i
        if np.any(mask):
            binned[i] = np.median(y[mask])

    finite = np.isfinite(binned)
    if np.sum(finite) >= 2:
        grid = np.arange(n_bins)
        binned[~finite] = np.interp(grid[~finite], grid[finite], binned[finite])
    elif np.any(finite):
        binned[~finite] = binned[finite][0]
    else:
        binned[:] = 1.0

    return binned
