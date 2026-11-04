# Dataset survey

Starting draft. Check every number on the dataset's own page, fill in the blanks, and add anything you find.

| Modality | Dataset | Diseases we use | Size | Patients known? | Licence | Our use | Kaggle / source |
|---|---|---|---|---|---|---|---|
| X-ray | NIH ChestX-ray14 | Pneumonia, Fibrosis, Emphysema (COPD proxy), Effusion; boxes for ~1,000 images | 112,120 images, 30,805 patients | Yes | CC0 (check) | Train / val / test, IoU evaluation | `nih-chest-xrays/data` |
| X-ray | NIH sample | Same as above | 5,606 images | Yes | CC0 (check) | Quick experiments | `nih-chest-xrays/sample` |
| X-ray | TB Chest X-ray Database | Tuberculosis vs normal | | | | Train / val / test | `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` |
| X-ray | Shenzhen | Tuberculosis vs normal | | | | Train / val / test | `raddar/tuberculosis-chest-xrays-shenzhen` |
| X-ray | Montgomery | Tuberculosis vs normal | | | | Train / val / test | `raddar/tuberculosis-chest-xrays-montgomery` |
| X-ray | IN-CXR (ICMR-NIRT) | Tuberculosis | | | Request access | External test set | ICMR-NIRT |
| CT | Large COVID-19 CT slice dataset | COVID-19, community-acquired pneumonia, normal | | | | Train / val / test | `maedemaftouni/large-covid19-ct-slice-dataset` |
| CT | OSIC Pulmonary Fibrosis Progression | Fibrosis (DICOM scans) | | Yes (one folder per patient) | Competition rules | Train / val / test | Kaggle competition `osic-pulmonary-fibrosis-progression` |

## Notes for the report

- **Why MRI is excluded:** almost no public, labelled lung MRI data exists for these diseases, and MRI isn't used to diagnose them in practice. Listed as future work.
- **Unknown labels:** each dataset only labels some diseases. Cells for diseases a dataset doesn't label are stored as -1 and ignored by the loss.
- **Source bias:** OSIC has only fibrosis patients. Normal CTs come from other datasets, so a model could learn the source instead of the disease. We check heatmaps and report results per source.
- **UK data (NCCID):** not used; access is restricted. Documented as a limitation.
