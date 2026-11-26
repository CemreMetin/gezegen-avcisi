import os
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "data/tess_tce_training.npz"
OUTPUT_PATH = "docs/triage_cheat_sheet.png"


def _cleanliness_score(l_view):
    """Lokal gorunumde dis bolge gurultusu dusuk ve ortada net cukuru olan ornekleri one cikarir."""
    edges = np.concatenate([l_view[:15], l_view[-15:]])
    center = l_view[24:37]
    edge_std = float(np.std(edges)) + 1e-6
    dip = float(np.median(edges) - np.min(center))
    return dip / edge_std


def make_cheat_sheet(n_per_side=4):
    if not os.path.exists(DATASET_PATH):
        print(f"Bulunamadi: {DATASET_PATH} (once python -m ai.prepare_data calistirin)")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data = np.load(DATASET_PATH)
    l_views = data["local_views"]
    scalars = data["scalars"]
    labels = data["labels"]
    tic_ids = data["tic_ids"]

    pc_idx = np.where(labels == 1.0)[0]
    fp_idx = np.where(labels == 0.0)[0]

    # En net 4 gercek gezegeni sec (gozun temiz U-sekline alissin)
    pc_ranked = sorted(pc_idx, key=lambda i: _cleanliness_score(l_views[i]), reverse=True)
    pc_pick = pc_ranked[:n_per_side]

    # Cesitli 4 yanlis alarm sec (bazi net V-seklinde, bazilari gurultulu)
    fp_ranked = sorted(fp_idx, key=lambda i: _cleanliness_score(l_views[i]), reverse=True)
    step = max(1, len(fp_ranked) // (n_per_side + 1))
    fp_pick = [fp_ranked[min(k * step, len(fp_ranked) - 1)] for k in range(n_per_side)]

    x = np.linspace(-2, 2, 61)
    fig, axes = plt.subplots(n_per_side, 2, figsize=(13, 10), facecolor="#121212")
    fig.suptitle(
        "TESS Planet Hunter — Visual Triage Cheat Sheet (Real NASA TESS Examples)",
        color="#ffffff",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for row in range(n_per_side):
        # Sol sutun: Gercek Gezegenler (PC)
        ax_l = axes[row, 0]
        i_pc = pc_pick[row]
        P, depth, Rp, Rstar, Teff, dur_h = scalars[i_pc]
        ax_l.set_facecolor("#1a1a1a")
        ax_l.axhline(0.0, color="#444444", linestyle="--", lw=0.8)
        ax_l.axvline(0.0, color="#555555", linestyle=":", lw=0.9)
        ax_l.plot(x, l_views[i_pc], "s-", color="#39FF14", ms=3.5, lw=1.3)
        ax_l.grid(True, color="#2a2a2a", linestyle="--")
        ax_l.tick_params(colors="#888888", labelsize=8)
        ax_l.set_title(
            f"REAL PLANET (Press Right)  |  TIC {tic_ids[i_pc]}  |  P={P:.2f}d  Rp={Rp:.2f} Rj  ({depth*1e6:.0f} ppm)",
            color="#39FF14",
            fontsize=9,
            loc="left",
        )

        # Sag sutun: Yanlis Alarmlar (FP)
        ax_r = axes[row, 1]
        i_fp = fp_pick[row]
        P2, depth2, Rp2, Rstar2, Teff2, dur_h2 = scalars[i_fp]
        ax_r.set_facecolor("#1a1a1a")
        ax_r.axhline(0.0, color="#444444", linestyle="--", lw=0.8)
        ax_r.axvline(0.0, color="#555555", linestyle=":", lw=0.9)
        ax_r.plot(x, l_views[i_fp], "o-", color="#FF5555", ms=3.5, lw=1.3)
        ax_r.grid(True, color="#2a2a2a", linestyle="--")
        ax_r.tick_params(colors="#888888", labelsize=8)
        ax_r.set_title(
            f"FALSE ALARM (Press Down)  |  TIC {tic_ids[i_fp]}  |  P={P2:.2f}d  Rp={Rp2:.2f} Rj  ({depth2*1e6:.0f} ppm)",
            color="#FF6B6B",
            fontsize=9,
            loc="left",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_PATH, dpi=220, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Kopya kagidi hazir: {OUTPUT_PATH}")


if __name__ == "__main__":
    make_cheat_sheet()
