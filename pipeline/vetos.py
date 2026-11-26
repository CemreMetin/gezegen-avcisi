import numpy as np


def run_analytical_vetos(time, flux, period, t0, duration, r_star):
    """Tek-cift derinlik farki (Odd-Even), gezegen yaricapi ve ikincil tutulma vetolari."""
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    half_dur_phase = (duration / 2.0) / period

    transit_num = np.round((time - t0) / period)
    odd_mask = (transit_num % 2 != 0) & (np.abs(phase) < half_dur_phase)
    even_mask = (transit_num % 2 == 0) & (np.abs(phase) < half_dur_phase)

    # Faz sinirinda (+0.5 / -0.5) sarilan ikincil tutulmayi tam yakala
    sec_mask = np.abs(np.abs(phase) - 0.5) < half_dur_phase
    out_mask = (np.abs(phase) > 2.0 * half_dur_phase) & (~sec_mask)

    if np.sum(odd_mask) < 3 or np.sum(even_mask) < 3:
        return False, "Yetersiz gecis noktasi", {}

    baseline = float(np.median(flux[out_mask])) if np.any(out_mask) else 1.0
    depth_odd = baseline - float(np.median(flux[odd_mask]))
    depth_even = baseline - float(np.median(flux[even_mask]))

    err_odd = float(np.std(flux[odd_mask])) / np.sqrt(np.sum(odd_mask))
    err_even = float(np.std(flux[even_mask])) / np.sqrt(np.sum(even_mask))
    denom = np.sqrt(err_odd**2 + err_even**2)
    z_odd_even = abs(depth_odd - depth_even) / denom if denom > 0 else 0.0

    avg_depth = max(0.0, (depth_odd + depth_even) / 2.0)
    r_planet_jup = (r_star * np.sqrt(avg_depth)) / 0.10045 if r_star > 0 else 0.0

    base_metrics = {
        "depth": float(avg_depth),
        "rp": float(r_planet_jup),
        "z_odd_even": float(z_odd_even),
    }

    if avg_depth <= 0.0:
        return False, "Pozitif gecis derinligi yok", base_metrics

    if z_odd_even > 3.0:
        return False, f"Cift Yildiz (Odd-Even Z={z_odd_even:.2f})", {**base_metrics, "type": "EB"}

    if r_planet_jup > 2.0:
        return False, f"Yildiz Boyutu (Rp={r_planet_jup:.2f} R_Jup)", {**base_metrics, "type": "EB"}

    if np.sum(sec_mask) >= 3:
        depth_sec = baseline - float(np.median(flux[sec_mask]))
        err_sec = float(np.std(flux[sec_mask])) / np.sqrt(np.sum(sec_mask))
        if depth_sec > 0.5 * avg_depth and err_sec > 0 and depth_sec > 3.0 * err_sec:
            return False, "Ikincil Tutulma Saptandi", {**base_metrics, "type": "EB"}

    return True, "Gecti", base_metrics
