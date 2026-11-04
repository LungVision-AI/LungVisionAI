"""Build a manifest from any dataset laid out as one folder per class.

Most Kaggle TB, pneumonia and CT datasets look like this:
    <root>/Normal/xxx.png
    <root>/Tuberculosis/yyy.png

--map tells the script what each folder means:
    folder=Label     images in this folder are positive for Label
    folder=normal    images in this folder are negative for every covered label

--covered lists the labels this dataset can speak about. Every other label is
written as -1 (unknown). If you leave it out, it is the set of labels in --map.

Patient IDs: many public sets have several images per patient. If the patient ID
is part of the filename, pass --patient-regex with one capture group, for example
--patient-regex "^(\\d+)_" for files like 00012_003.png. Without it, each image is
treated as its own patient (the script warns you, because that can leak patients
across splits).

Example (Kaggle TB dataset):
    python -m lungvision.data.prepare_folder --root /kaggle/input/tb-chest-xray \
        --config configs/xray.yaml --source tb_kaggle --modality xray \
        --map "Tuberculosis=Tuberculosis" --map "Normal=normal" \
        --out data/manifests/tb_kaggle.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from lungvision.config import load_config
from lungvision.data.manifest import list_images, save_manifest, summarize


def parse_map(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--map expects folder=Label, got '{item}'")
        folder, label = item.split("=", 1)
        out[folder.strip()] = label.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", required=True, help="Short dataset name, e.g. tb_kaggle")
    ap.add_argument("--modality", required=True, choices=["xray", "ct"])
    ap.add_argument("--map", action="append", required=True, help="folder=Label or folder=normal (repeat)")
    ap.add_argument("--covered", nargs="*", default=None)
    ap.add_argument("--patient-regex", default=None)
    ap.add_argument("--patient-from-parent", action="store_true",
                    help="Use the image's parent folder name as the patient ID (CT slices per scan)")
    ap.add_argument("--label-from-filename", default=None,
                    help="Regex with one group taken from the filename and looked up in --map instead of "
                         "the folder name. Shenzhen/Montgomery: CHNCXR_0001_0.png (0=normal, 1=TB) -> "
                         "--label-from-filename '_(\\d)\\.png$' --map 1=Tuberculosis --map 0=normal")
    ap.add_argument("--normal-negative-for", nargs="*", default=[],
                    help="Extra labels for which images in a 'normal' folder count as negative (0). Example: "
                         "healthy CT slices are negatives for Fibrosis even though this dataset doesn't label fibrosis.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels = cfg["labels"]
    fmap = parse_map(args.map)
    unknown = {v for v in fmap.values() if v != "normal" and v not in labels}
    if unknown:
        raise SystemExit(f"Labels {unknown} are not in the config's label list {labels}")
    covered = set(args.covered) if args.covered else {v for v in fmap.values() if v != "normal"}
    pat_re = re.compile(args.patient_regex) if args.patient_regex else None
    name_re = re.compile(args.label_from_filename) if args.label_from_filename else None

    root = Path(args.root)
    rows, skipped = [], 0
    for img in list_images(root):
        if name_re:
            m = name_re.search(img.name)
            cls = m.group(1) if m and m.group(1) in fmap else None
        else:
            rel = img.relative_to(root).parts
            cls = next((p for p in rel[:-1] if p in fmap), None)
        if cls is None:
            skipped += 1
            continue
        target = fmap[cls]

        if args.patient_from_parent:
            pid = img.parent.name
        elif pat_re:
            m = pat_re.search(img.name)
            pid = m.group(1) if m else img.stem
        else:
            pid = img.stem

        row = {"path": str(img), "patient_id": f"{args.source}_{pid}",
               "source": args.source, "modality": args.modality}
        for lab in labels:
            if lab in covered:
                row[lab] = int(target == lab)
            elif target == "normal" and lab in args.normal_negative_for:
                row[lab] = 0
            else:
                row[lab] = -1
        rows.append(row)

    if not rows:
        raise SystemExit("No images matched. Check --root and the folder names in --map.")
    df = pd.DataFrame(rows)
    if skipped:
        print(f"Skipped {skipped:,} images in folders not listed in --map")
    if not pat_re and not args.patient_from_parent:
        print("Warning: no patient ID rule given, so each image counts as its own patient.")
    print(summarize(df, labels).to_string(index=False))
    save_manifest(df, args.out)


if __name__ == "__main__":
    main()
