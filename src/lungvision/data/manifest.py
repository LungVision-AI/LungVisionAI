"""Shared helpers for manifests.

A manifest is a CSV with one row per image:
    path, patient_id, source, modality, [split], <one column per label>

Label values:
    1  = disease present
    0  = disease absent
   -1  = unknown (this source never labels that disease)

The -1 value matters because we mix datasets. The NIH set has no TB labels and
the TB sets have no fibrosis labels, so those cells are "unknown" and the loss
ignores them instead of treating them as negatives.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_COLS = ["path", "patient_id", "source", "modality"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def empty_row(labels: list[str]) -> dict:
    return {lab: -1 for lab in labels}


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} rows -> {path}")


def summarize(df: pd.DataFrame, labels: list[str], by: str | None = None) -> pd.DataFrame:
    """Count positives / negatives / unknowns per label (optionally per split or source)."""
    groups = [("all", df)] if by is None else list(df.groupby(by))
    rows = []
    for name, g in groups:
        for lab in labels:
            if lab not in g:
                continue
            rows.append({
                "group": name,
                "label": lab,
                "positive": int((g[lab] == 1).sum()),
                "negative": int((g[lab] == 0).sum()),
                "unknown": int((g[lab] == -1).sum()),
            })
    return pd.DataFrame(rows)


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
