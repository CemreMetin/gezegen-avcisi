import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from ai.model import AstroNet1D, fit_scalar_stats, normalize_scalars
from ai.augmentation import LightCurveAugmentor

DATASET_PATH = "data/tess_tce_training.npz"
SPLITS_PATH = "data/tess_tce_splits.npz"
WEIGHTS_PATH = "ai/weights/astronet_tess.pt"


class TCEDataset(Dataset):
    def __init__(self, g, l, s, y, mean, scale, augment=False):
        self.g = g
        self.l = l
        self.s = s
        self.y = y
        self.mean = mean
        self.scale = scale
        self.aug = LightCurveAugmentor() if augment else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        g, l, s = self.g[idx], self.l[idx], self.s[idx]
        if self.aug is not None:
            g, l, s = self.aug(g, l, s)
        s_norm = normalize_scalars(s, self.mean, self.scale)
        return (
            torch.tensor(g, dtype=torch.float32).unsqueeze(0),
            torch.tensor(l, dtype=torch.float32).unsqueeze(0),
            torch.tensor(s_norm, dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0),
        )


def make_splits(labels, seed=42):
    """Veri sizintisini onlemek icin %70 egitim, %15 dogrulama, %15 test ayirir."""
    idx_all = np.arange(len(labels))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(idx_all, labels))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(sss2.split(temp_idx, labels[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    return train_idx, val_idx, test_idx


def train_astronet(epochs=35, batch_size=32, lr=1e-3):
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SPLITS_PATH), exist_ok=True)

    data = np.load(DATASET_PATH)
    g = data["global_views"]
    l = data["local_views"]
    s = data["scalars"]
    y = data["labels"]

    train_idx, val_idx, test_idx = make_splits(y.astype(int))
    np.savez(SPLITS_PATH, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    # Scaler YALNIZCA egitim setine fit edilir (veri sizintisi sifir)
    mean, scale = fit_scalar_stats(s[train_idx])

    train_ds = TCEDataset(g[train_idx], l[train_idx], s[train_idx], y[train_idx], mean, scale, augment=True)
    val_ds = TCEDataset(g[val_idx], l[val_idx], s[val_idx], y[val_idx], mean, scale, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    torch.manual_seed(42)
    model = AstroNet1D()
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_auc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    print(
        f"AstroNet 1D Egitimi Basliyor ({epochs} Epoch | Parametre={n_params:,} | "
        f"Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)})..."
    )
    for epoch in range(epochs):
        model.train()
        loss_total = 0.0
        for bg, bl, bs, by in train_loader:
            optimizer.zero_grad()
            preds = model(bg, bl, bs)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_total += loss.item()

        model.eval()
        val_true, val_prob = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for bg, bl, bs, by in val_loader:
                probs = model(bg, bl, bs)
                val_loss_total += criterion(probs, by).item()
                val_prob.extend(probs.squeeze(-1).numpy().tolist())
                val_true.extend(by.squeeze(-1).numpy().tolist())

        val_loss = val_loss_total / max(len(val_loader), 1)
        val_auc = roc_auc_score(val_true, val_prob) if len(set(val_true)) > 1 else 0.5
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d} | Train Loss: {loss_total / max(len(train_loader), 1):.4f} "
                f"| Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
            )

    torch.save(
        {
            "model_state_dict": best_state,
            "scaler_mean": mean,
            "scaler_scale": scale,
            "best_val_auc": float(best_auc),
        },
        WEIGHTS_PATH,
    )
    print(f"En iyi model kaydedildi: {WEIGHTS_PATH} (Val Loss: {best_val_loss:.4f} | Val AUC: {best_auc:.4f})")


if __name__ == "__main__":
    train_astronet()
