import os
import sys
import glob
import datetime
import numpy as np
import torch
from ai.model import load_checkpoint, normalize_scalars

PENDING_DIR = "pending_candidates"
WEIGHTS_PATH = "ai/weights/astronet_tess.pt"


def score_batches(npz_paths=None, threshold=0.40, weights_path=WEIGHTS_PATH):
    if not npz_paths:
        npz_paths = sorted(glob.glob(os.path.join(PENDING_DIR, "batch_*.npz")))
    if not npz_paths:
        print("Puanlanacak batch_*.npz dosyasi bulunamadi.")
        return None

    model, mean, scale = load_checkpoint(weights_path, map_location="cpu")

    all_tics, all_g, all_l, all_s, all_sde = [], [], [], [], []
    for path in npz_paths:
        data = np.load(path, allow_pickle=True)
        if len(data["tic_ids"]) == 0:
            continue
        all_tics.append(data["tic_ids"])
        all_g.append(data["global_views"])
        all_l.append(data["local_views"])
        all_s.append(data["scalars"])
        if "sde" in data:
            all_sde.append(data["sde"])
        else:
            all_sde.append(np.zeros(len(data["tic_ids"]), dtype=np.float32))

    if not all_tics:
        print("Batch dosyalarinda aday bulunamadi.")
        return None

    tic_ids = np.concatenate(all_tics)
    g_views = np.concatenate(all_g, axis=0)
    l_views = np.concatenate(all_l, axis=0)
    scalars = np.concatenate(all_s, axis=0)
    sdes = np.concatenate(all_sde, axis=0)

    s_norm = normalize_scalars(scalars, mean, scale)

    with torch.no_grad():
        g_t = torch.tensor(g_views, dtype=torch.float32).unsqueeze(1)
        l_t = torch.tensor(l_views, dtype=torch.float32).unsqueeze(1)
        s_t = torch.tensor(s_norm, dtype=torch.float32)
        scores = model(g_t, l_t, s_t).squeeze(-1).numpy().astype(np.float32)

    keep = scores >= threshold
    total = len(tic_ids)
    kept = int(np.sum(keep))
    print(f"Toplam {total} aday puanlandi | Esik (>={threshold:.2f}) gecen: {kept}")

    if kept == 0:
        return None

    order = np.argsort(-scores[keep])
    out_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(PENDING_DIR, f"scored_{out_stamp}.npz")

    np.savez_compressed(
        out_path,
        tic_ids=tic_ids[keep][order],
        global_views=g_views[keep][order],
        local_views=l_views[keep][order],
        scalars=scalars[keep][order],
        scores=scores[keep][order],
        sde=sdes[keep][order],
    )
    print(f"Triyaja hazir paket kaydedildi: {out_path}")
    print(f"Incelemek icin: python triage.py {out_path}")
    return out_path


if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else None
    score_batches(paths)
