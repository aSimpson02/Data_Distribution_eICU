#!/usr/bin/env bash
set -euo pipefail

# Make sure Python can import src/*
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# Use your venv python explicitly
PY="$(pwd)/.venv/Scripts/python.exe"

# Keep Windows consoles sane
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 LANG=C.UTF-8 LC_ALL=C.UTF-8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

SCENARIOS=("random" "temporal" "hospital")
LABEL="hospital_mortality"
SPLITS_DIR="data/csv_splits"
ICL_ROOT="results/icl"
ICL_PREDS_DIR="${ICL_ROOT}/preds"
FIGS_CAL_DIR="results/figs/calibration_icl"
FIGS_CMP_DIR="results/figs/compare_icl"

mkdir -p "$ICL_ROOT" "$ICL_PREDS_DIR" "$FIGS_CAL_DIR" "$FIGS_CMP_DIR"

echo "[1/4] ICL: Prototype (baseline) ..."
for s in "${SCENARIOS[@]}"; do
  "$PY" scripts/icl_prototype.py \
    --scenario "$s" \
    --label_col "$LABEL" \
    --splits_dir "$SPLITS_DIR" \
    --results_dir "$ICL_ROOT" \
    --cap_train 20000 \
    --k 50 --metric euclidean --weight distance \
    --calib none platt \
    --seed 42
done

echo "[2/4] ICL: Prototype++ (PCA + weighted + softmax) ..."
for s in "${SCENARIOS[@]}"; do
  "$PY" scripts/icl_prototype_plus.py \
    --scenario "$s" \
    --label_col "$LABEL" \
    --splits_dir "$SPLITS_DIR" \
    --results_dir "$ICL_ROOT" \
    --cap_train 50000 \
    --k 64 --metric cosine --weight softmax --temperature 0.1 \
    --feature_weight l1 --repr pca --pca_components 64 \
    --calib none platt \
    --seed 42
done

echo "[3/4] Consolidate ICL scores ..."
# Build results/icl/all_scores.csv from the 1-row metrics files
"$PY" - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("results/icl")
rows=[]
for f in sorted(root.glob("*.csv")):
    if f.name.startswith(("random_","temporal_","hospital_")):
        df = pd.read_csv(f)
        if len(df)==1 and {"auroc","ece"}.issubset(df.col
