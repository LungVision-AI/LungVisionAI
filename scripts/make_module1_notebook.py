"""Generates notebooks/module1_data_preparation.ipynb (run once; the notebook is committed)."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# LungVision AI · Module 1: Data Collection and Preparation

This notebook completes Module 1 on the real datasets:

| Sub-task | What happens here |
|---|---|
| 1.1 | Find every attached dataset and count its images |
| 1.2 | Datasets are attached as Kaggle inputs (no download needed on Kaggle) |
| 1.3 | Build a label list (manifest) for each dataset |
| 1.4 | Convert OSIC 3D CT scans (DICOM) to 2D lung-window slices |
| 1.5 | Remove duplicate images across datasets |
| 1.6 | Split train / validation / test by patient |
| 1.7 | Dataset statistics report and figures |

**Before you click Run All**

1. **Settings → Internet → On.** (No GPU needed for Module 1.)
2. **Add Input** (right panel → *Add Input*), search and add:
   - `nih-chest-xrays/data` (NIH ChestX-ray14, full) **or** `nih-chest-xrays/sample` (5,606 images, for a quick first run; then set `QUICK = True` below)
   - `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`
   - `raddar/tuberculosis-chest-xrays-shenzhen`
   - `raddar/tuberculosis-chest-xrays-montgomery`
   - `maedemaftouni/large-covid19-ct-slice-dataset`
   - Competition data `osic-pulmonary-fibrosis-progression` (open the competition page and accept the rules first)
3. **Code:** the notebook clones the project from GitHub. If the Module 1 code isn't pushed yet, upload the `lungvision` folder as a Kaggle dataset named `lungvision-code` and add it as an input instead.

Any dataset you don't add is simply skipped. Full run time: roughly 30–60 minutes (mostly hashing the NIH images).
""")

code(r"""
import os, re, sys, glob, json, shutil, subprocess
from pathlib import Path

QUICK = False            # True = use the NIH 5,606-image sample (nih-chest-xrays/sample)
OSIC_EVERY_NTH = 4       # keep every Nth CT slice from the middle of each OSIC scan
CT_PATIENT_REGEX = None  # leave None: built-in rules read patient IDs from the CT file names (see note in 1.3)
GITHUB_REPO = "https://github.com/LungVision-AI/LungVisionAI"

INPUT = Path(os.environ.get("LV_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("LV_WORK", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)

def run(cmd, cwd=None):
    print("$", cmd)
    p = subprocess.run(cmd, shell=True, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-6000:])
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")
    return p.stdout

# ---- get the code ----
CODE = WORK / "lungvision"
local = os.environ.get("LV_CODE") or next((str(Path(p).parent) for p in glob.glob(str(INPUT / "**/lungvision/pyproject.toml"), recursive=True)), None)
if CODE.exists():
    shutil.rmtree(CODE)   # always start from the attached / latest code, never a stale copy
if local:
    skip_top = lambda d, names: [n for n in names if Path(d) == Path(local) and (n in ("data", "runs", "results") or n.startswith("deploy"))]
    shutil.copytree(local, CODE, ignore=skip_top)
    print("Copied code from", local)
else:
    run(f"git clone --depth 1 {GITHUB_REPO} {CODE}")
if not (CODE / "src/lungvision/data/build_split.py").exists():
    raise SystemExit("The Module 1 code is not in the repository yet. Push it to GitHub, or add the lungvision-code dataset.")
run(f"{sys.executable} -m pip install -q -e {CODE} imagehash tabulate pydicom pylibjpeg pylibjpeg-libjpeg")
os.chdir(CODE)
CFG_X, CFG_CT = "configs/xray.yaml", "configs/ct.yaml"
MAN = Path("data/manifests"); MAN.mkdir(parents=True, exist_ok=True)
""")

md("## 1.1 Find the attached datasets")

code(r"""
IMG = {".png", ".jpg", ".jpeg"}
dirs_with_images, csvs, dicom_parents = {}, {}, set()
for root, dnames, fnames in os.walk(INPUT):
    if "lungvision" in Path(root).parts:
        continue
    n = 0
    for f in fnames:
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG: n += 1
        elif ext == ".csv": csvs.setdefault(f, os.path.join(root, f))
        elif ext == ".dcm": dicom_parents.add(str(Path(root).parent))
    if n: dirs_with_images[root] = n

def child_dirs(parent):
    try: return [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
    except OSError: return []

def find_class_root(want):
    # A folder whose sub-folders match every keyword group in `want`, e.g. {"tb": ["tuberculosis"], "normal": ["normal"]}
    best = None
    for d in sorted({str(Path(p).parent) for p in dirs_with_images} | {str(Path(p).parent.parent) for p in dirs_with_images}):
        kids = child_dirs(d)
        found = {}
        for key, kws in want.items():
            for k in kids:
                low = k.lower().replace("-", "").replace("_", "")
                if any(low == kw or low.endswith(kw) or kw in low for kw in kws) and k not in found.values():
                    found[key] = k; break
        if len(found) == len(want):
            best = (d, found); break
    return best

found = {}
# NIH
nih_csv = csvs.get("sample_labels.csv" if QUICK else "Data_Entry_2017.csv")
if nih_csv: found["nih"] = {"csv": nih_csv, "bbox": csvs.get("BBox_List_2017.csv")}
# TB database (folders Normal / Tuberculosis)
tb = find_class_root({"Tuberculosis": ["tuberculosis"], "normal": ["normal"]})
if tb: found["tb_kaggle"] = {"root": tb[0], "map": {tb[1]["Tuberculosis"]: "Tuberculosis", tb[1]["normal"]: "normal"}}
# Shenzhen / Montgomery: label in the file name
for name, prefix in (("shenzhen", "CHNCXR_"), ("montgomery", "MCUCXR_")):
    d = next((p for p in dirs_with_images if glob.glob(os.path.join(p, prefix + "*"))), None)
    if d: found[name] = {"root": d, "prefix": prefix}
# CT slice dataset: non-COVID / COVID / CAP folders
ct = find_class_root({"normal": ["noncovid", "normal"], "COVID-19": ["covid"], "Pneumonia": ["cap", "pneumonia"]})
if ct: found["ct_covid"] = {"root": ct[0], "map": {v: k for k, v in ct[1].items()}}
# OSIC: train/<patient>/<n>.dcm
osic = next((p for p in sorted(dicom_parents) if Path(p).name == "train"), None)
if osic: found["osic"] = {"root": osic}

print(f"{'dataset':12s} details")
for k, v in found.items():
    print(f"{k:12s} {v}")
missing = {"nih", "tb_kaggle", "shenzhen", "montgomery", "ct_covid", "osic"} - set(found)
if missing: print("\nNot attached (skipped):", ", ".join(sorted(missing)))
print("\nFolders with the most images:")
for d, n in sorted(dirs_with_images.items(), key=lambda t: -t[1])[:12]:
    print(f"  {n:7,d}  {d}")
""")

md("""
## 1.2 Datasets

On Kaggle the datasets are attached, so there is nothing to download. Outside Kaggle, `bash scripts/download_data.sh all` downloads the same datasets with the Kaggle API.

## 1.3 Build a manifest (label list) for each dataset
""")

code(r"""
xray_parts, ct_parts = [], []
py = sys.executable

if "nih" in found:
    v = found["nih"]
    nih_root = INPUT / Path(v["csv"]).relative_to(INPUT).parts[0]   # the whole NIH dataset folder
    run(f'{py} -m lungvision.data.prepare_nih --root "{nih_root}" '
        f'--csv "{v["csv"]}" --config {CFG_X} --out {MAN}/nih.csv')
    xray_parts.append(f"{MAN}/nih.csv")

if "tb_kaggle" in found:
    v = found["tb_kaggle"]; maps = " ".join(f'--map "{k}={t}"' for k, t in v["map"].items())
    run(f'{py} -m lungvision.data.prepare_folder --root "{v["root"]}" --config {CFG_X} --source tb_kaggle '
        f'--modality xray {maps} --out {MAN}/tb_kaggle.csv')
    xray_parts.append(f"{MAN}/tb_kaggle.csv")

for name in ("shenzhen", "montgomery"):
    if name in found:
        v = found[name]
        run(f'{py} -m lungvision.data.prepare_folder --root "{v["root"]}" --config {CFG_X} --source {name} --modality xray '
            f'--label-from-filename "_(\\d)\\.png$" --map 1=Tuberculosis --map 0=normal '
            f'--patient-regex "^({v["prefix"]}\\d+)" --out {MAN}/{name}.csv')
        xray_parts.append(f"{MAN}/{name}.csv")

if "ct_covid" in found:
    v = found["ct_covid"]
    sample = sorted(os.listdir(os.path.join(v["root"], next(iter(v["map"])))))[:15]
    print("Example CT file names:", sample)
    maps = " ".join(f'--map "{k}={t}"' for k, t in v["map"].items())
    pat = f'--patient-regex "{CT_PATIENT_REGEX}"' if CT_PATIENT_REGEX else ""
    # Healthy (non-COVID) slices are also negatives for Fibrosis; otherwise Fibrosis would have no negatives at all.
    run(f'{py} -m lungvision.data.prepare_folder --root "{v["root"]}" --config {CFG_CT} --source ct_covid '
        f'--modality ct {maps} {pat} --normal-negative-for Fibrosis --out {MAN}/ct_covid.csv')
    if not CT_PATIENT_REGEX:
        run(f'{py} -m lungvision.data.ct_patients {MAN}/ct_covid.csv')   # group slices by patient / scan
    ct_parts.append(f"{MAN}/ct_covid.csv")
""")

md("""
**Note on CT patient IDs.** CT slices from the same patient must stay in the same split, or test accuracy is inflated. The CT dataset merges several sources that name files differently (`P104_37.png`, `16_Morozov_study_0001_12.png`, `6_Rahimzadeh_normal5_patient219_...`). `lungvision.data.ct_patients` reads the patient or scan ID from each format. For sources with no patient ID at all (Bjorke, Cohen), it keeps the whole source together. The line `patient IDs: 16,833 -> ~870` above shows it worked.

## 1.4 Convert OSIC CT scans (3D DICOM) to 2D slices
""")

code(r"""
if "osic" in found:
    out = Path("data/ct_slices/Fibrosis")
    if not out.exists() or not any(out.iterdir()):
        run(f'{py} -m lungvision.data.ct_to_slices --input "{found["osic"]["root"]}" --format dicom '
            f'--out {out} --every {OSIC_EVERY_NTH}')
    run(f'{py} -m lungvision.data.prepare_folder --root data/ct_slices --config {CFG_CT} --source osic --modality ct '
        f'--map "Fibrosis=Fibrosis" --covered Fibrosis --patient-from-parent --out {MAN}/osic.csv')
    ct_parts.append(f"{MAN}/osic.csv")
""")

md("## 1.5 + 1.6 Remove duplicates and split by patient")

code(r"""
if xray_parts:
    ext = "--external incxr" if any("incxr" in p for p in xray_parts) else ""
    run(f'{py} -m lungvision.data.build_split --config {CFG_X} --inputs {" ".join(xray_parts)} {ext} '
        f'--workers 8 --out {MAN}/xray_split.csv')
if ct_parts:
    run(f'{py} -m lungvision.data.build_split --config {CFG_CT} --inputs {" ".join(ct_parts)} '
        f'--workers 8 --out {MAN}/ct_split.csv')
if xray_parts and ct_parts:
    run(f'{py} -m lungvision.data.build_modality --xray {MAN}/xray_split.csv --ct {MAN}/ct_split.csv '
        f'--out {MAN}/modality_split.csv')
""")

md("## 1.7 Dataset statistics report")

code(r"""
args = []
if (MAN / "xray_split.csv").exists(): args.append(f"--xray {MAN}/xray_split.csv")
if (MAN / "ct_split.csv").exists(): args.append(f"--ct {MAN}/ct_split.csv")
run(f"{py} scripts/module1_report.py {' '.join(args)} --out reports/module1")
try:
    from IPython.display import Markdown, Image, display
    display(Markdown(Path("reports/module1/data_stats.md").read_text().split("![")[0]))
    for f in sorted(Path("reports/module1").glob("fig_*.png")):
        display(Image(str(f), width=720))
except ImportError:
    print(Path("reports/module1/data_stats.md").read_text())
""")

md("""
## Save the outputs

Everything Module 1 produced is zipped into `module1_outputs.zip` in the notebook's **Output** tab. Download it, then commit `reports/module1/` to GitHub; that is the Module 1 deliverable for the guide.

The manifests point to `/kaggle/input/...` paths. Use them in later Kaggle notebooks (training) with the same datasets attached. Kaggle also keeps this notebook's output, so later notebooks can add it as an input.
""")

code(r"""
bundle = WORK / "module1_outputs"
if bundle.exists(): shutil.rmtree(bundle)
bundle.mkdir()
shutil.copytree("data/manifests", bundle / "manifests")
shutil.copytree("reports/module1", bundle / "reports_module1")
shutil.make_archive(str(WORK / "module1_outputs"), "zip", bundle)
print("Saved", WORK / "module1_outputs.zip")
for f in sorted(bundle.rglob("*")):
    if f.is_file(): print(f"  {f.relative_to(bundle)}  ({f.stat().st_size/1024:.0f} KB)")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "module1_data_preparation.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
