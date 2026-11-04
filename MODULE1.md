# Module 1: Data Collection and Preparation

Status: **complete** (run on the real datasets on Kaggle, 04-11-2026).

## What it does
1. Finds the attached datasets (NIH ChestX-ray14, TB Chest X-ray Database, Shenzhen, Montgomery, Large COVID-19 CT slice dataset, OSIC Pulmonary Fibrosis).
2. Builds a label list (manifest) for each dataset: 1 = disease present, 0 = absent, -1 = not labelled by that dataset.
3. Converts OSIC 3D CT scans (DICOM) to 2D lung-window slices.
4. Removes exact and near-identical duplicate images, carrying their labels over to the kept copy.
5. Splits train / validation / test (80/10/10) by patient, with an automatic leak check.
6. Writes the dataset statistics report (`reports/module1/data_stats.md`).

## Results
| | Images | Patients | Train / Val / Test | Patients in more than one split |
|---|---|---|---|---|
| Chest X-ray | 113,598 | 32,308 | 91,853 / 10,911 / 10,834 | 0 |
| CT slices | 21,751 | 1,049 | 17,184 / 2,370 / 2,197 | 0 |

3,915 duplicate images removed. 3,487 "Normal" images in the TB Chest X-ray Database were found to be copies of NIH images.

## How to run
Open `notebooks/module1_data_preparation.ipynb` on Kaggle, attach the datasets listed at its top, turn Internet on and click Run All (about 1 hour on CPU).

## Code
- `src/lungvision/data/`: manifests, CT conversion, CT patient IDs, duplicate removal and split
- `scripts/module1_report.py`: statistics report
- `configs/`: disease labels and settings for X-ray and CT
