#!/usr/bin/env bash
# Download LungVision datasets from Kaggle with the Kaggle command-line tool.
#
# You do NOT need this on Kaggle notebooks: there you click "Add Input" and the data
# appears under /kaggle/input. Use this script on Colab, your laptop or a lab machine.
#
# One-time setup:
#   1. Make a Kaggle account and verify your phone number.
#   2. kaggle.com -> your profile -> Settings -> API -> "Create New Token".
#      This downloads kaggle.json.
#   3. mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#      (Colab: upload kaggle.json first, then run the same commands.)
#   4. For OSIC only: open the competition page once and click "Join / I understand and accept"
#      https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression/rules
#
# Usage:
#   bash scripts/download_data.sh nih-sample tb          # small start (good for testing)
#   bash scripts/download_data.sh nih tb shenzhen montgomery ct-covid osic
#   bash scripts/download_data.sh all
#
# Everything goes into data/raw/<name>/. Check free disk space first: the full NIH
# set is very large (tens of GB). Start with nih-sample if space or bandwidth is tight.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/raw
[ $# -eq 0 ] && { sed -n '2,25p' "$0"; exit 0; }

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Installing the Kaggle CLI ..."
  pip install -q kaggle
fi
if [ ! -f ~/.kaggle/kaggle.json ] && [ -z "${KAGGLE_USERNAME:-}" ] && [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  echo "No Kaggle API token found. See the setup steps at the top of this script." >&2
  exit 1
fi

dl_dataset() {  # name slug
  local name=$1 slug=$2
  if [ -d "$OUT/$name" ] && [ -n "$(ls -A "$OUT/$name" 2>/dev/null)" ]; then
    echo "== $name already downloaded, skipping"; return
  fi
  echo "== Downloading $name ($slug)"
  mkdir -p "$OUT/$name"
  kaggle datasets download -d "$slug" -p "$OUT/$name" --unzip
  show "$name"
}

dl_competition() {  # name slug
  local name=$1 slug=$2
  if [ -d "$OUT/$name" ] && [ -n "$(ls -A "$OUT/$name" 2>/dev/null)" ]; then
    echo "== $name already downloaded, skipping"; return
  fi
  echo "== Downloading competition $name ($slug)"
  mkdir -p "$OUT/$name"
  kaggle competitions download -c "$slug" -p "$OUT/$name"
  (cd "$OUT/$name" && for z in *.zip; do unzip -q "$z" && rm "$z"; done)
  show "$name"
}

show() {  # print the top of the folder tree so you know the folder names for --map
  echo "-- $OUT/$1:"
  find "$OUT/$1" -maxdepth 3 -type d | head -20
  echo "   $(find "$OUT/$1" -type f | wc -l) files"
}

get() {
  case $1 in
    nih)         dl_dataset nih         nih-chest-xrays/data ;;
    nih-sample)  dl_dataset nih_sample  nih-chest-xrays/sample ;;
    tb)          dl_dataset tb_kaggle   tawsifurrahman/tuberculosis-tb-chest-xray-dataset ;;
    shenzhen)    dl_dataset shenzhen    raddar/tuberculosis-chest-xrays-shenzhen ;;
    montgomery)  dl_dataset montgomery  raddar/tuberculosis-chest-xrays-montgomery ;;
    ct-covid)    dl_dataset ct_covid    maedemaftouni/large-covid19-ct-slice-dataset ;;
    osic)        dl_competition osic    osic-pulmonary-fibrosis-progression ;;
    all)         for d in nih tb shenzhen montgomery ct-covid osic; do get $d; done ;;
    *) echo "Unknown dataset '$1'. Choose: nih nih-sample tb shenzhen montgomery ct-covid osic all" >&2; exit 1 ;;
  esac
}

for d in "$@"; do get "$d"; done

cat <<'EOF'

Done. Not on Kaggle (request access separately):
  - IN-CXR (ICMR-NIRT): Indian TB X-rays, best used as the external test set.
  - MosMedData CT (mosmed.ai): optional extra COVID-19 CT scans in NIfTI format.
Next: build manifests (see README "Running on real data"). Use the folder names printed
above in the --map arguments.
EOF
