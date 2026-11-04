"""Convert 3D CT scans (DICOM folders or NIfTI files) into 2D PNG slices.

The models work on 2D images, so each CT scan becomes a set of axial slices with
a lung window applied (level -600 HU, width 1500 HU). Only the middle part of the
scan is kept by default, since the top and bottom slices rarely show lung.

Output layout keeps one folder per scan, which lets prepare_folder.py use
--patient-from-parent so all slices of a patient stay in the same split:
    <out>/<class>/<scan_id>/slice_000.png

Examples:
    # OSIC fibrosis: one DICOM folder per patient
    python -m lungvision.data.ct_to_slices --input /kaggle/input/osic-pulmonary-fibrosis-progression/train \
        --format dicom --out data/ct_slices/Fibrosis --every 4

    # NIfTI volumes (e.g. MosMedData)
    python -m lungvision.data.ct_to_slices --input data/raw/mosmed/CT-1 --format nifti \
        --out data/ct_slices/COVID-19

Needs pydicom (DICOM) or nibabel (NIfTI): pip install pydicom nibabel
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

WINDOW_LEVEL, WINDOW_WIDTH = -600, 1500


def window(hu: np.ndarray) -> np.ndarray:
    lo, hi = WINDOW_LEVEL - WINDOW_WIDTH / 2, WINDOW_LEVEL + WINDOW_WIDTH / 2
    img = np.clip(hu, lo, hi)
    return ((img - lo) / (hi - lo) * 255).astype(np.uint8)


def load_dicom_series(folder: Path) -> np.ndarray:
    import pydicom

    slices = []
    for f in folder.iterdir():
        try:
            ds = pydicom.dcmread(str(f))
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        slope = float(getattr(ds, "RescaleSlope", 1))
        inter = float(getattr(ds, "RescaleIntercept", 0))
        z = float(getattr(ds, "ImagePositionPatient", [0, 0, getattr(ds, "InstanceNumber", 0)])[2])
        slices.append((z, arr * slope + inter))
    slices.sort(key=lambda t: t[0])
    if not slices:
        return np.zeros((0, 1, 1), np.float32)
    shape = slices[0][1].shape
    return np.stack([s for _, s in slices if s.shape == shape])


def load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib

    vol = nib.load(str(path)).get_fdata().astype(np.float32)  # (x, y, z)
    return np.transpose(vol, (2, 1, 0))  # -> (z, y, x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--format", required=True, choices=["dicom", "nifti"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=3, help="Keep every Nth slice")
    ap.add_argument("--keep-middle", type=float, default=0.6, help="Fraction of slices kept around the middle")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    src, out = Path(args.input), Path(args.out)
    if args.format == "dicom":
        scans = [(d.name, d) for d in sorted(src.iterdir()) if d.is_dir()]
    else:
        scans = [(p.name.split(".")[0], p) for p in sorted(src.iterdir()) if p.name.endswith((".nii", ".nii.gz"))]

    total = 0
    for scan_id, path in scans:
        vol = load_dicom_series(path) if args.format == "dicom" else load_nifti(path)
        n = len(vol)
        if n == 0:
            print(f"Skipping {scan_id}: no readable slices")
            continue
        half = int(n * args.keep_middle / 2)
        mid = n // 2
        idx = range(max(0, mid - half), min(n, mid + half + 1), args.every)
        d = out / scan_id
        d.mkdir(parents=True, exist_ok=True)
        for i in idx:
            img = cv2.resize(window(vol[i]), (args.size, args.size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(d / f"slice_{i:03d}.png"), img)
            total += 1
    print(f"Wrote {total:,} slices from {len(scans)} scans to {out}")


if __name__ == "__main__":
    main()
