"""Patient / scan IDs for the Large COVID-19 CT slice dataset (Maftouni et al.).

The dataset merges slices from several public sources, and each source names its
files differently. Slices of one patient must stay in one split, so we read the
patient (or scan) out of the file name. When a source gives no patient ID at all,
we keep the whole source together, which is the safe choice against leakage.

    16_Morozov_study_0001_12.png              -> 16_Morozov_study_0001
    14_Jun_coronacases_case8_101.png          -> 14_Jun_coronacases_case8
    P104_37.png / N027_83.png / cap006_10.png -> P104 / N027 / cap006
    normal027_120.png                         -> N027   (same numbering as N027; kept together to be safe)
    6_Rahimzadeh_normal5_patient219_SR_4_IM00012.png -> 6_Rahimzadeh_normal5_patient219
    17_Zhao_2020.03.21.20040691-p18-67-7.png  -> 17_Zhao_2020.03.21.20040691   (one paper's figures)
    15_Cohen_..., 13_Bjorke_12.png            -> 15_Cohen, 13_Bjorke          (no patient ID: whole source)
"""
from __future__ import annotations

import os
import re

_RULES = [
    (re.compile(r"^(6_Rahimzadeh_.*?patient\d+)"), None),
    (re.compile(r"^(17_Zhao_.+?)-p\d+"), None),
    (re.compile(r"^(17_Zhao)_"), None),
    (re.compile(r"^(15_Cohen)_"), None),
    (re.compile(r"^(13_Bjorke)_"), None),
    (re.compile(r"^normal(\d+)_\d+\.\w+$"), lambda m: f"N{m.group(1)}"),
    (re.compile(r"^(.+)_\d+\.\w+$"), None),
]


def maftouni_patient_id(path: str) -> str:
    name = os.path.basename(path)
    for rx, fmt in _RULES:
        m = rx.search(name)
        if m:
            return fmt(m) if fmt else m.group(1)
    return os.path.splitext(name)[0]


def main() -> None:
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser(description="Rewrite patient_id in a ct_covid manifest using the file-name rules.")
    ap.add_argument("manifest")
    ap.add_argument("--source", default="ct_covid")
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    before = df["patient_id"].nunique()
    df["patient_id"] = f"{args.source}_" + df["path"].map(maftouni_patient_id)
    df.to_csv(args.manifest, index=False)
    print(f"patient IDs: {before:,} -> {df['patient_id'].nunique():,} (slices of one patient now share an ID)")


if __name__ == "__main__":
    main()
