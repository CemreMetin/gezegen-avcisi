import numpy as np
import torch
import torch.nn as nn

LOG_SCALAR_INDICES = (0, 1, 2, 3, 5)  # period, depth, rp_jup, r_star, duration_h


def preprocess_scalars(scalars):
    """Carpik pozitif skalerlere log1p uygular (Teff lineer kalir)."""
    arr = np.asarray(scalars, dtype=np.float32).copy()
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    for idx in LOG_SCALAR_INDICES:
        arr[:, idx] = np.log1p(np.maximum(arr[:, idx], 0.0))
    return arr[0] if single else arr


def fit_scalar_stats(scalars):
    proc = preprocess_scalars(scalars)
    mean = np.mean(proc, axis=0).astype(np.float32)
    scale = np.std(proc, axis=0).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return mean, scale


def normalize_scalars(scalars, mean, scale):
    proc = preprocess_scalars(scalars)
    return ((proc - mean) / scale).astype(np.float32)


class AstroNet1D(nn.Module):
    """Az veriyle ezber (overfitting) yapmayan kompakt cift-kollu 1D CNN."""

    def __init__(self):
        super().__init__()
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
            nn.Conv1d(16, 24, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
            nn.Conv1d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Hem ortalama sekli hem keskin gecis cukurunu koruyan kompakt ozet (24*4*2 = 192 yerine 24*4 = 96)
        self.pool_avg = nn.AdaptiveAvgPool1d(4)
        self.pool_max = nn.AdaptiveMaxPool1d(4)

        self.scalar_fc = nn.Sequential(
            nn.Linear(6, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # global: 24*4*2=192, local: 24*4*2=192, scalar: 16 -> toplam 400 (eski: 2096)
        total_dense = 192 + 192 + 16
        self.classifier = nn.Sequential(
            nn.Linear(total_dense, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _pool_branch(self, x):
        a = self.pool_avg(x).flatten(1)
        m = self.pool_max(x).flatten(1)
        return torch.cat([a, m], dim=1)

    def forward(self, g, l, s):
        out_g = self._pool_branch(self.global_conv(g))
        out_l = self._pool_branch(self.local_conv(l))
        out_s = self.scalar_fc(s)
        merged = torch.cat([out_g, out_l, out_s], dim=1)
        return self.classifier(merged)


def load_checkpoint(weights_path="ai/weights/astronet_tess.pt", map_location="cpu"):
    ckpt = torch.load(weights_path, map_location=map_location, weights_only=False)
    model = AstroNet1D()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    else:
        model.load_state_dict(ckpt)
        mean = np.zeros(6, dtype=np.float32)
        scale = np.ones(6, dtype=np.float32)
    model.eval()
    return model, mean, scale
