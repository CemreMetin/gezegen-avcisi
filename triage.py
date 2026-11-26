import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["toolbar"] = "None"


class CandidateTriage:
    def __init__(self, npz_path, log_dir="triage_logs"):
        self.npz_path = npz_path
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        data = np.load(npz_path, allow_pickle=True)
        self.tic_ids = data["tic_ids"]
        self.global_views = data["global_views"]
        self.local_views = data["local_views"]
        self.scalars = data["scalars"]
        self.scores = data["scores"] if "scores" in data else np.full(len(self.tic_ids), np.nan)
        self.sdes = data["sde"] if "sde" in data else np.full(len(self.tic_ids), np.nan)

        self.total = len(self.tic_ids)
        self.idx = 0
        self.decisions = {}  # idx -> record dict

        self.setup_ui()
        self.update_plot()

    def setup_ui(self):
        self.fig = plt.figure(figsize=(13, 6), facecolor="#121212")
        self.fig.canvas.manager.set_window_title(f"Triyaj - {os.path.basename(self.npz_path)}")

        self.ax_global = self.fig.add_axes([0.06, 0.12, 0.42, 0.70], facecolor="#1e1e1e")
        self.ax_local = self.fig.add_axes([0.54, 0.12, 0.42, 0.70], facecolor="#1e1e1e")

        self.hud = self.fig.text(0.06, 0.88, "", color="#00F0FF", fontsize=10.5, family="monospace")
        self.fig.text(
            0.06,
            0.03,
            "[Sag] Onayla  |  [Asagi] Ele  |  [Yukari] Supheli  |  [Sol] Geri Al  |  [q] Kaydet & Cikis",
            color="#888888",
            fontsize=9,
            family="monospace",
        )

        self.ax_global.axvline(0.0, color="#555555", linestyle=":", lw=1)
        self.ax_global.axhline(0.0, color="#444444", linestyle="--", lw=0.8)
        self.ax_local.axvline(0.0, color="#555555", linestyle=":", lw=1)
        self.ax_local.axhline(0.0, color="#444444", linestyle="--", lw=0.8)

        self.line_global, = self.ax_global.plot(np.linspace(-0.5, 0.5, 201), np.zeros(201), "o", color="#00F0FF", ms=3)
        self.line_local, = self.ax_local.plot(np.linspace(-2, 2, 61), np.zeros(61), "s-", color="#39FF14", ms=4, lw=1.2)

        self.ax_global.set_title("Global Gorunum (Tam Yorunge)", color="#cccccc", fontsize=10)
        self.ax_local.set_title("Lokal Gorunum (Yakin Gecis)", color="#cccccc", fontsize=10)

        for ax in [self.ax_global, self.ax_local]:
            ax.grid(True, color="#2a2a2a", linestyle="--")
            ax.tick_params(colors="#888888")

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def update_plot(self):
        if self.idx >= self.total:
            self.hud.set_text("TUM ADAYLAR TARANDI. Kararlar kaydedildi — kapatmak icin [q]'ya bas.")
            self.fig.canvas.draw_idle()
            return

        tic = self.tic_ids[self.idx]
        g_flux = self.global_views[self.idx]
        l_flux = self.local_views[self.idx]
        P, depth, Rp, Rstar, Teff, dur_h = self.scalars[self.idx]
        score = self.scores[self.idx]
        score_str = f"%{score * 100:.1f}" if np.isfinite(score) else "N/A"

        self.hud.set_text(
            f"TIC {tic} | AI Skor: {score_str} | P: {P:.3f}g | Sure: {dur_h:.2f}s | Derinlik: {depth * 1e6:.0f} ppm\n"
            f"Rp: {Rp:.2f} R_Jup | R*: {Rstar:.2f} R_Sun | Teff: {Teff:.0f} K | Ilerleme: {self.idx + 1}/{self.total}"
        )

        self.line_global.set_ydata(g_flux)
        g_pad = max(0.15, (float(np.max(g_flux)) - float(np.min(g_flux))) * 0.15)
        self.ax_global.set_ylim(float(np.min(g_flux)) - g_pad, float(np.max(g_flux)) + g_pad)

        self.line_local.set_ydata(l_flux)
        l_pad = max(0.15, (float(np.max(l_flux)) - float(np.min(l_flux))) * 0.15)
        self.ax_local.set_ylim(float(np.min(l_flux)) - l_pad, float(np.max(l_flux)) + l_pad)

        self.fig.canvas.draw_idle()

    def record_decision(self, label):
        if self.idx >= self.total:
            return
        P, depth, Rp, Rstar, Teff, dur_h = self.scalars[self.idx]
        score = float(self.scores[self.idx]) if np.isfinite(self.scores[self.idx]) else -1.0
        self.decisions[self.idx] = {
            "tic_id": str(self.tic_ids[self.idx]),
            "decision": label,
            "ai_score": round(score, 4),
            "period_days": round(float(P), 5),
            "depth_ppm": round(float(depth * 1e6), 1),
            "rp_jup": round(float(Rp), 3),
            "source_batch": os.path.basename(self.npz_path),
        }
        self.idx += 1
        self.update_plot()

    def undo(self):
        if self.idx <= 0:
            return
        self.idx -= 1
        self.decisions.pop(self.idx, None)
        self.update_plot()

    def save_decisions(self):
        if not self.decisions:
            return
        batch_name = os.path.basename(self.npz_path)
        for label in ("verified", "rejected", "flagged"):
            rows = [r for r in self.decisions.values() if r["decision"] == label]
            if not rows:
                continue
            csv_path = os.path.join(self.log_dir, f"{label}_candidates.csv")
            new_df = pd.DataFrame(rows)
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                old_df = pd.read_csv(csv_path, dtype={"tic_id": str})
                # Ayni batch+tic_id varsa guncelle (cift kaydi ve undo artiklarini onler)
                if "source_batch" in old_df.columns:
                    keep_mask = ~(
                        (old_df["source_batch"] == batch_name)
                        & (old_df["tic_id"].isin(new_df["tic_id"]))
                    )
                    old_df = old_df[keep_mask]
                out_df = pd.concat([old_df, new_df], ignore_index=True)
            else:
                out_df = new_df
            out_df.to_csv(csv_path, index=False)
        print(f"Triyaj kararlari kaydedildi ({len(self.decisions)} aday) -> {self.log_dir}/")

    def on_close(self, _event):
        self.save_decisions()

    def on_key(self, event):
        if event.key == "right":
            self.record_decision("verified")
        elif event.key == "down":
            self.record_decision("rejected")
        elif event.key == "up":
            self.record_decision("flagged")
        elif event.key == "left":
            self.undo()
        elif event.key in ["q", "escape"]:
            self.save_decisions()
            plt.close(self.fig)
            sys.exit(0)


def find_latest_npz():
    for pattern in ("pending_candidates/scored_*.npz", "pending_candidates/batch_*.npz"):
        files = sorted(glob.glob(pattern))
        if files:
            return files[-1]
    return None


if __name__ == "__main__":
    target_npz = sys.argv[1] if len(sys.argv) >= 2 else find_latest_npz()
    if not target_npz or not os.path.exists(target_npz):
        print("Kullanim: python triage.py [pending_candidates/scored_xxx.npz]")
        print("pending_candidates/ icinde henuz paket bulunamadi.")
        sys.exit(1)
    CandidateTriage(target_npz)
    plt.show()
