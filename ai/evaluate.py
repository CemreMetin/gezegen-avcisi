import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from ai.model import load_checkpoint
from ai.train import TCEDataset, DATASET_PATH, SPLITS_PATH, WEIGHTS_PATH


def load_test_dataset(npz_path=DATASET_PATH, splits_path=SPLITS_PATH, weights_path=WEIGHTS_PATH):
    model, mean, scale = load_checkpoint(weights_path, map_location="cpu")
    data = np.load(npz_path)

    if os.path.exists(splits_path):
        idx = np.load(splits_path)["test_idx"]
        split_name = "TEST (Gorulmemis Veri)"
    else:
        idx = np.arange(len(data["labels"]))
        split_name = "TUM VERI (Uyari: splits bulunamadi)"

    ds = TCEDataset(
        data["global_views"][idx],
        data["local_views"][idx],
        data["scalars"][idx],
        data["labels"][idx],
        mean,
        scale,
        augment=False,
    )
    return model, ds, split_name


def evaluate(threshold=0.50):
    model, dataset, split_name = load_test_dataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    y_true, y_prob = [], []
    with torch.no_grad():
        for g, l, s, y in loader:
            probs = model(g, l, s)
            y_prob.extend(probs.squeeze(-1).numpy().tolist())
            y_true.extend(y.squeeze(-1).numpy().tolist())

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pc_mean = float(np.mean(y_prob[y_true == 1])) if np.any(y_true == 1) else 0.0
    fp_mean = float(np.mean(y_prob[y_true == 0])) if np.any(y_true == 0) else 0.0

    print("=" * 60)
    print(f"Degerlendirme Kumesi : {split_name} (n={len(y_true)})")
    print(f"Dogruluk (Accuracy)  : %{accuracy_score(y_true, y_pred) * 100:.2f}")
    print(f"Kesinlik (Precision) : %{precision_score(y_true, y_pred, zero_division=0) * 100:.2f}")
    print(f"Duyarlilik (Recall)  : %{recall_score(y_true, y_pred, zero_division=0) * 100:.2f}")
    print(f"ROC-AUC Skoru        : {roc_auc_score(y_true, y_prob):.4f}")
    print("-" * 60)
    print(f"Gercek Gezegenlere (PC) Ortalama Skor : %{pc_mean * 100:.1f}")
    print(f"Yanlis Alarmlara   (FP) Ortalama Skor : %{fp_mean * 100:.1f}")
    print(f"Karisiklik Matrisi -> Dogru PC: {tp} | Kacan PC: {fn} | Dogru FP: {tn} | Yanlis Alarm: {fp}")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=["FP", "PC"], zero_division=0))


if __name__ == "__main__":
    evaluate()
