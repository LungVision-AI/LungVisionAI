"""Build a manifest from NIH ChestX-ray14.

Expects the Kaggle layout:
    <root>/Data_Entry_2017.csv
    <root>/images_001/images/00000001_000.png ... (any sub-folders are fine)

Works for the full set, the resized version and the 5,606-image sample
(sample_labels.csv), as long as you point --csv at the right file.

Example:
    python -m lungvision.data.prepare_nih --root /kaggle/input/data \
        --config configs/xray.yaml --out data/manifests/nih.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from lungvision.config import load_config
from lungvision.data.manifest import IMAGE_EXTS, save_manifest, summarize

# NIH label names that LungVision maps onto its own label names.
# Emphysema is the closest NIH label to COPD.
NIH_TO_LV = {
    "Pneumonia": "Pneumonia",
    "Fibrosis": "Fibrosis",
    "Emphysema": "Emphysema",
    "Effusion": "Effusion",
    "Consolidation": "Consolidation",
    "Infiltration": "Infiltration",
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pneumothorax": "Pneumothorax",
    "Edema": "Edema",
    "Pleural_Thickening": "Pleural_Thickening",
    "Hernia": "Hernia",
}


def index_images(root: Path) -> dict[str, str]:
    index = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                index[f] = os.path.join(dirpath, f)
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Folder holding the NIH csv and image folders")
    ap.add_argument("--csv", default=None, help="Label csv (default: <root>/Data_Entry_2017.csv)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="nih")
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels = cfg["labels"]
    root = Path(args.root)
    csv_path = Path(args.csv) if args.csv else root / "Data_Entry_2017.csv"
    meta = pd.read_csv(csv_path)

    print("Indexing image files ...")
    index = index_images(root)
    print(f"Found {len(index):,} image files")

    covered = {NIH_TO_LV[k] for k in NIH_TO_LV} & set(labels)
    missing_files = 0
    rows = []
    meta.columns = [c.strip() for c in meta.columns]
    for rec in meta.to_dict("records"):
        path = index.get(rec["Image Index"])
        if path is None:
            missing_files += 1
            continue
        findings = set(str(rec["Finding Labels"]).split("|"))
        row = {
            "path": path,
            "patient_id": f"{args.source}_{int(rec['Patient ID'])}",
            "source": args.source,
            "modality": "xray",
            "view": rec.get("View Position", ""),
        }
        for lab in labels:
            if lab in covered:
                nih_names = [k for k, v in NIH_TO_LV.items() if v == lab]
                row[lab] = int(any(n in findings for n in nih_names))
            else:
                row[lab] = -1
        rows.append(row)

    df = pd.DataFrame(rows)
    if missing_files:
        print(f"Warning: {missing_files:,} rows in the csv had no image file and were skipped")
    print(f"Labels covered by NIH: {sorted(covered)}; set to unknown: {sorted(set(labels) - covered)}")
    print(summarize(df, labels).to_string(index=False))
    save_manifest(df, args.out)


if __name__ == "__main__":
    main()
