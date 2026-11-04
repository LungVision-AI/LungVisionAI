"""One-off fix for Module 1 outputs made before labels were carried over from duplicates.

The TB database's "Normal" images are copies of NIH images. Duplicate removal kept the
NIH copy and dropped the TB copy, and with it the label Tuberculosis = 0. This script
re-hashes only the dropped TB-database images, finds the kept copy of each one in
xray_split.csv (same perceptual hash, or within 8 of 256 bits), and copies the TB label
onto it where the kept copy has none. The split itself is unchanged.

Paste into a Kaggle cell (same inputs attached), or run:
    python scripts/fix_tb_labels.py --xray xray_split.csv --tb tb_kaggle.csv --out xray_split_fixed.csv
"""
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor

import imagehash
import numpy as np
import pandas as pd
from PIL import Image


def find(name, roots=("/kaggle/working", "/kaggle/input")):
    for r in roots:
        hits = sorted(glob.glob(f"{r}/**/{name}", recursive=True))
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"{name} not found under {roots}")


def phash(path):
    with Image.open(path) as im:
        return str(imagehash.phash(im.convert("L"), hash_size=16))


def main(xray, tb, out, max_dist=8, workers=8):
    x = pd.read_csv(xray)
    t = pd.read_csv(tb)
    removed = t[~t["path"].isin(set(x["path"]))].reset_index(drop=True)
    print(f"xray_split: {len(x):,} rows | TB database: {len(t):,} images, {len(removed):,} removed as duplicates")

    with ThreadPoolExecutor(workers) as ex:
        hashes = list(ex.map(phash, removed["path"]))

    kept = np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in x["phash"]])
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    filled = unmatched = already = 0
    matched_sources = {}
    for h, label in zip(hashes, removed["Tuberculosis"]):
        q = np.frombuffer(bytes.fromhex(h), dtype=np.uint8)
        d = pop[np.bitwise_xor(kept, q)].sum(axis=1, dtype=np.int32)
        j = int(d.argmin())
        if d[j] > max_dist:
            unmatched += 1
            continue
        src = x.at[j, "source"]
        matched_sources[src] = matched_sources.get(src, 0) + 1
        if x.at[j, "Tuberculosis"] == -1:
            x.at[j, "Tuberculosis"] = int(label)
            filled += 1
        else:
            already += 1

    print(f"Kept copies found: {len(removed) - unmatched:,} (in {matched_sources}); "
          f"TB label added: {filled:,}; already labelled: {already:,}; no match: {unmatched:,}")
    print(x.groupby("split")["Tuberculosis"].value_counts().unstack().rename(columns={-1: "not labelled", 0: "negative", 1: "positive"}))
    x.to_csv(out, index=False)
    print("Saved", out)
    return x


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xray", default=None)
    ap.add_argument("--tb", default=None)
    ap.add_argument("--out", default="/kaggle/working/xray_split_fixed.csv")
    a = ap.parse_args()
    main(a.xray or find("xray_split.csv"), a.tb or find("tb_kaggle.csv"), a.out)
