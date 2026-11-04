"""Merge manifests, remove duplicate images, and split by patient.

Steps:
1. Concatenate the per-dataset manifests.
2. Hash every image (perceptual hash). Images with the same hash are duplicates;
   one copy is kept. Public lung datasets often re-share each other's images,
   so this stops the same picture appearing in both train and test.
3. Split by patient, so no patient appears in more than one split. Splits are
   stratified by (source, main label) to keep class balance similar.
4. Any source passed to --external is held out entirely as an external test set,
   for checking how the model does on a hospital/dataset it never saw.

Example:
    python -m lungvision.data.build_split --config configs/xray.yaml \
        --inputs data/manifests/nih.csv data/manifests/tb_kaggle.csv data/manifests/incxr.csv \
        --external incxr --out data/manifests/xray_split.csv
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from lungvision.config import load_config
from lungvision.data.manifest import save_manifest, summarize


def phash(path: str) -> str:
    import imagehash

    try:
        with Image.open(path) as im:
            return str(imagehash.phash(im.convert("L"), hash_size=16))
    except Exception:
        return f"unreadable::{path}"


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def near_duplicate_groups(hashes: list[str], max_dist: int) -> np.ndarray:
    """Group images whose perceptual hashes differ in at most `max_dist` bits.

    Multi-index hashing: the 256-bit hash is cut into 16 chunks of 16 bits. Two
    hashes within 15 bits of each other must agree on at least one chunk, so we
    only compare images that share a chunk value. Returns a group id per image."""
    n = len(hashes)
    packed = np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hashes])
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    n_chunks = packed.shape[1] // 2
    for c in range(n_chunks):
        key = packed[:, 2 * c].astype(np.int32) << 8 | packed[:, 2 * c + 1]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            idx = order[s:e]
            if len(idx) > 3000:  # a very common chunk value; compare in slices to bound memory
                idx_slices = [idx[k:k + 3000] for k in range(0, len(idx), 3000)]
            else:
                idx_slices = [idx]
            for sl in idx_slices:
                block = packed[sl]
                for a in range(len(sl) - 1):
                    d = _POP[np.bitwise_xor(block[a + 1:], block[a])].sum(1)
                    for b in np.flatnonzero(d <= max_dist):
                        ra, rb = find(sl[a]), find(sl[a + 1 + b])
                        if ra != rb:
                            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def merge_labels(df: pd.DataFrame, grp: pd.Series, drop: np.ndarray, labels: list[str]) -> int:
    """Before dropping duplicates, copy their known labels onto the copy that is kept.

    Example: the TB database's "Normal" images are copies of NIH images. The NIH copy is
    kept (it has the NIH disease labels) and gains the TB database's label Tuberculosis = 0,
    which NIH never recorded. Only unknown (-1) cells on the kept copy are filled; a known
    value is never overwritten. Returns the number of cells filled."""
    if not drop.any():
        return 0
    keep_idx = pd.Series(df.index, index=df.index).groupby(grp.values).transform("first")
    targets = keep_idx[drop].to_numpy()
    filled = 0
    for lab in labels:
        donors = pd.Series(df.loc[drop, lab].to_numpy(), index=targets)
        best = donors.groupby(level=0).max()          # 1 beats 0 beats -1
        cur = df.loc[best.index, lab]
        fill = (cur.to_numpy() == -1) & (best.to_numpy() >= 0)
        df.loc[best.index[fill], lab] = best.to_numpy()[fill]
        filled += int(fill.sum())
    return filled


def dedup(df: pd.DataFrame, workers: int, max_dist: int, rehash: bool = False,
          labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels or []
    if "phash" in df and not rehash and df["phash"].notna().all():
        print(f"Reusing perceptual hashes already in the manifests for {len(df):,} images")
    else:
        print(f"Hashing {len(df):,} images ...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            df = df.assign(phash=list(ex.map(phash, df["path"], chunksize=64)))
    bad = df["phash"].astype(str).str.startswith("unreadable::")
    if bad.any():
        print(f"Dropping {bad.sum():,} unreadable images")
        df = df[~bad]
    df = df.reset_index(drop=True)

    exact = df.duplicated("phash", keep="first")
    n = merge_labels(df, df["phash"], exact.to_numpy(), labels)
    print(f"Exact duplicates removed: {int(exact.sum()):,} (labels carried over to the kept copy: {n:,})")
    df = df[~exact].reset_index(drop=True)

    if max_dist > 0 and len(df) > 1:
        grp = pd.Series(near_duplicate_groups(df["phash"].tolist(), max_dist))
        # Only near-duplicates from a DIFFERENT patient are removed: they are copies of one image
        # re-shared under another name/dataset and would leak across splits. Near-identical images
        # of the same patient (neighbouring CT slices, follow-up X-rays) stay; the patient-level
        # split already keeps them together.
        first_patient = df["patient_id"].groupby(grp).transform("first")
        near = (grp.duplicated(keep="first") & (df["patient_id"] != first_patient)).to_numpy()
        cross = df.loc[near, "source"] != df["source"].groupby(grp).transform("first")[near]
        n = merge_labels(df, grp, near, labels)
        print(f"Near-duplicates removed (hash distance <= {max_dist} of 256 bits, different patient): "
              f"{int(near.sum()):,} ({int(cross.sum()):,} of them copies from another dataset; "
              f"labels carried over to the kept copy: {n:,})")
        df = df[~near].reset_index(drop=True)
    return df


def strat_key(df: pd.DataFrame, labels: list[str]) -> pd.Series:
    """Main label per image: first positive label in config order, else 'normal'."""
    lab = df[labels].to_numpy()
    first = np.where((lab == 1).any(1), (lab == 1).argmax(1), -1)
    names = np.array(labels + ["normal"])
    return df["source"].astype(str) + "|" + names[first]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--external", nargs="*", default=[], help="Source names to hold out as external test")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--near-dup", type=int, default=8,
                    help="Also remove near-duplicates whose hashes differ in at most this many bits (0 = exact only)")
    ap.add_argument("--rehash", action="store_true", help="Recompute hashes even if the manifests already have them")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels = cfg["labels"]
    df = pd.concat([pd.read_csv(p) for p in args.inputs], ignore_index=True)
    for lab in labels:
        if lab not in df:
            df[lab] = -1
    df[labels] = df[labels].fillna(-1).astype(int)
    df["patient_id"] = df["patient_id"].astype(str)
    print(f"Merged {len(df):,} images from {df['source'].nunique()} sources")

    if not args.no_dedup:
        df = dedup(df, args.workers, args.near_dup, args.rehash, labels)

    df["split"] = ""
    ext = df["source"].isin(args.external)
    df.loc[ext, "split"] = "external"
    inner = df[~ext]

    # One fold = 1/n of patients. Pick n so the test and val sizes come out right.
    n_folds = max(3, int(round(1 / min(args.val, args.test))))
    n_test = max(1, int(round(args.test * n_folds)))
    n_val = max(1, int(round(args.val * n_folds)))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    y = strat_key(inner, labels)
    fold_of = pd.Series(-1, index=inner.index)
    for k, (_, idx) in enumerate(sgkf.split(inner, y, groups=inner["patient_id"])):
        fold_of.iloc[idx] = k
    split = np.where(fold_of < n_test, "test", np.where(fold_of < n_test + n_val, "val", "train"))
    df.loc[inner.index, "split"] = split

    # Safety check: no patient in two splits.
    per_patient = df.groupby("patient_id")["split"].nunique()
    leaks = per_patient[per_patient > 1]
    if len(leaks):
        raise SystemExit(f"Patient leak detected for {len(leaks)} patients, e.g. {leaks.index[:5].tolist()}")
    print("Patient-level check passed: no patient appears in more than one split.")

    print(df["split"].value_counts().to_string())
    print(summarize(df, labels, by="split").to_string(index=False))
    save_manifest(df, args.out)


if __name__ == "__main__":
    main()
