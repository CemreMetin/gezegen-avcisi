import os
import argparse
import numpy as np
from astroquery.mast import Observations

QUEUE_PATH = "data/tic_queue.txt"
CURSOR_PATH = "data/cursor.txt"


def build_spoc_queue(target_count=10000, sectors=None, reset_cursor=True):
    """MAST uzerinden gercekten TESS SPOC isik egrisi bulunan yildizlarin TIC listesini uretir."""
    os.makedirs(os.path.dirname(QUEUE_PATH), exist_ok=True)
    if sectors is None:
        sectors = list(range(1, 30))

    collected = []
    seen = set()

    print(f"MAST TESS SPOC hedef kuyrugu olusturuluyor (Hedef: {target_count} benzersiz TIC)...")
    for sec in sectors:
        if len(collected) >= target_count:
            break
        try:
            obs = Observations.query_criteria(
                obs_collection="TESS",
                provenance_name="SPOC",
                dataproduct_type="timeseries",
                sequence_number=sec,
            )
            if len(obs) == 0:
                continue

            new_in_sec = 0
            for raw_id in obs["target_name"]:
                tic = str(raw_id).strip().lstrip("0")
                if tic.isdigit() and tic not in seen:
                    seen.add(tic)
                    collected.append(tic)
                    new_in_sec += 1
                    if len(collected) >= target_count:
                        break
            print(f"Sektor {sec:02d}: +{new_in_sec} yeni TIC (Toplam: {len(collected)}/{target_count})")
        except Exception as e:
            print(f"Sektor {sec:02d} uyarisi: {e}")

    rng = np.random.default_rng(42)
    rng.shuffle(collected)

    with open(QUEUE_PATH, "w") as f:
        for tic in collected:
            f.write(f"{tic}\n")

    if reset_cursor:
        with open(CURSOR_PATH, "w") as f:
            f.write("0\n")

    print(f"Tamamlandi: {len(collected)} SPOC TIC -> {QUEUE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TESS SPOC TIC kuyrugu olusturucu")
    parser.add_argument("--count", type=int, default=10000, help="Kuyruga eklenecek hedef sayisi")
    parser.add_argument("--keep-cursor", action="store_true", help="data/cursor.txt sifirlanmasin")
    args = parser.parse_args()
    build_spoc_queue(target_count=args.count, reset_cursor=not args.keep_cursor)
