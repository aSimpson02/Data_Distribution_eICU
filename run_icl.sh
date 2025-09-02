#!/usr/bin/env bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PY="$(pwd)/.venv/Scripts/python.exe"
set -euo pipefail

# UTF-8
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# ------------- Config -------------
SCENARIOS=("random" "temporal" "hospital")
LABEL="hospital_mortality"
SPLITS_DIR="data/csv_splits"

ICL_ROOT="results/icl"
ICL_PREDS_DIR="${ICL_ROOT}/preds"
FIGS_CAL_DIR="results/figs/calibration_icl"
FIGS_CMP_DIR="results/figs/compare_icl"
# ----------------------------------

mkdir -p "$ICL_ROOT" "$ICL_PREDS_DIR" "$FIGS_CAL_DIR" "$FIGS_CMP_DIR"

echo "[1/6] ICL: TabPFN ..."
for s in "${SCENARIOS[@]}"; do
  "$PY" scripts/icl_tabpfn.py \
    --scenario "$s" \
    --label_col "$LABEL" \
    --splits_dir "$SPLITS_DIR" \
    --results_dir "$ICL_ROOT" \
    --device cpu \
    --cap_train 4000 \
    --calib none platt \
    --seed 42
done

echo "[2/6] ICL: Prototype (baseline) ..."
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

echo "[3/6] ICL: Prototype++ (stronger, weighted + PCA) ..."
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
    --abstain_margin 0.00 \
    --seed 42
done

echo "[4/6] Consolidate ICL scores ..."
"$PY" - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("results/icl")
rows=[]
for f in root.glob("*.csv"):
    if f.name.startswith(("random_","temporal_","hospital_")) and f.name.endswith(".csv"):
        df = pd.read_csv(f)
        if len(df)==1 and {"auroc","ece"}.issubset(df.columns):
            rows.append(df)
if rows:
    all_scores = pd.concat(rows, ignore_index=True)
    (root/"all_scores.csv").write_text(all_scores.to_csv(index=False))
    print("[OK] results/icl/all_scores.csv", len(all_scores), "rows")
else:
    print("[WARN] no ICL score rows found")
PY

"$PY" scripts/make_compare_all_scores.py \
  --scores results/icl/all_scores.csv \
  --out    results/icl/compare_all_scores.csv

echo "[5/6] Calibration plots (ICL models) ..."
"$PY" scripts/make_calibration_plots.py \
  --splits random temporal hospital \
  --models tabpfn protoicl protoicl_plus \
  --tags none platt \
  --preds_dir "$ICL_PREDS_DIR" \
  --outdir "$FIGS_CAL_DIR"

echo "[6/6] Score comparison plots (ICL models) ..."
"$PY" scripts/plot_compare_all_scores.py \
  --in     results/icl/compare_all_scores.csv \
  --outdir "$FIGS_CMP_DIR" \
  --metrics auroc ece brier nll auprc \
  --model_order tabpfn protoicl protoicl_plus \
  --tags_pref notuned none platt icl

echo "ICL done. See:"
echo " - results/icl/*.csv (ICL metrics; compare_all_scores.csv)"
echo " - results/icl/preds/*.csv (ICL per-example preds)"
echo " - results/figs/calibration_icl/ (reliability curves)"
echo " - results/figs/compare_icl/ (grouped + delta charts)"
