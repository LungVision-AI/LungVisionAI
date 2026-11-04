"""Build the manifest for the modality checker (X-ray vs CT).

It reuses the X-ray and CT split manifests, so the modality model's train/val/test
splits line up with the disease models' splits. The single label is "CT":
1 for CT slices, 0 for X-rays.

Example:
    python -m lungvision.data.build_modality --xray data/manifests/xray_split.csv \
        --ct data/manifests/ct_split.csv --out data/manifests/modality_split.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from lungvision.data.manifest import save_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xray", required=True)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-class", type=int, default=20000,
                    help="Cap per modality per split; this task is easy and doesn't need everything")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    parts = []
    for path, is_ct in ((args.xray, 0), (args.ct, 1)):
        df = pd.read_csv(path)
        df = df[df["split"].isin(["train", "val", "test"])]
        df = pd.concat([g.sample(min(len(g), args.max_per_class), random_state=args.seed)
                        for _, g in df.groupby("split")])
        parts.append(df[["path", "patient_id", "source", "modality", "split"]].assign(CT=is_ct))
    out = pd.concat(parts, ignore_index=True)
    print(out.groupby(["split", "CT"]).size().to_string())
    save_manifest(out, args.out)


if __name__ == "__main__":
    main()
