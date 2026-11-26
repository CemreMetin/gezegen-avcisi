import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from ai.evaluate import load_test_dataset


def generate_curves(output_path="ai/reports/model_curves.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model, dataset, split_name = load_test_dataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    y_true, y_prob = [], []
    with torch.no_grad():
        for g, l, s, y in loader:
            probs = model(g, l, s)
            y_prob.extend(probs.squeeze(-1).numpy().tolist())
            y_true.extend(y.squeeze(-1).numpy().tolist())

    y_true, y_prob = np.array(y_true, dtype=int), np.array(y_prob, dtype=float)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#121212")
    fig.suptitle(f"AstroNet 1D Performansi — {split_name} (n={len(y_true)})", color="#ffffff", fontsize=12)
    for ax in (ax_roc, ax_pr):
        ax.set_facecolor("#1a1a1a")
        ax.grid(True, color="#2a2a2a", linestyle="--")
        ax.tick_params(colors="#888888")

    ax_roc.plot(fpr, tpr, color="#00F0FF", lw=2, label=f"AUC = {auc(fpr, tpr):.4f}")
    ax_roc.plot([0, 1], [0, 1], color="#666666", linestyle=":")
    ax_roc.set_xlabel("False Positive Rate", color="#cccccc")
    ax_roc.set_ylabel("True Positive Rate", color="#cccccc")
    ax_roc.set_title("ROC Curve", color="#ffffff")
    ax_roc.legend(facecolor="#121212", labelcolor="#ffffff")

    ax_pr.plot(recall, precision, color="#39FF14", lw=2, label=f"AP = {average_precision_score(y_true, y_prob):.4f}")
    ax_pr.set_xlabel("Recall", color="#cccccc")
    ax_pr.set_ylabel("Precision", color="#cccccc")
    ax_pr.set_title("Precision-Recall Curve", color="#ffffff")
    ax_pr.legend(facecolor="#121212", labelcolor="#ffffff")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Kaydedildi: {output_path}")


if __name__ == "__main__":
    generate_curves()
