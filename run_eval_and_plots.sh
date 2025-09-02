#!/usr/bin/env bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
set -euo pipefail

# Make Windows terminals behave
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# ----------------- Config -----------------
SCENARIOS=("random" "temporal" "hospital")
MODELS=("lr" "rf" "xgb" "lgbm")
CALIBS=("none" "platt")
SEEDS=(42)
LABEL="hospital_mortality"
SPLITS_DIR="data/csv_splits"
RESULTS_DIR="results"
# ------------------------------------------

echo "[1/6] Evaluating (no-tune) ..."
for s in "${SCENARIOS[@]}"; do
  python scripts/eval_no_tune.py \
    --scenario "$s" \
    --models "${MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --label_col "$LABEL" \
    --splits_dir "$SPLITS_DIR" \
    --results_dir "$RESULTS_DIR"
done

echo "[2/6] Tuning + calibration (val) ..."
for s in "${SCENARIOS[@]}"; do
  python scripts/tune_with_val.py \
    --scenario "$s" \
    --models "${MODELS[@]}" \
    --calib "${CALIBS[@]}" \
    --seeds "${SEEDS[@]}" \
    --label_col "$LABEL" \
    --splits_dir "$SPLITS_DIR" \
    --results_dir "$RESULTS_DIR"
done

echo "[3/6] Consolidating results ..."
python scripts/compare_results.py || true
python scripts/make_compare_all_scores.py

echo "[4/6] Calibration plots from per-example predictions ..."
python scripts/make_calibration_plots.py --all

echo "[5/6] Score comparison plots (grouped + deltas) ..."
python scripts/plot_compare_all_scores.py --metrics auroc ece brier nll auprc

echo "[6/6] Shift diagnostics + LaTeX tables ..."
python scripts/shift_diagnostics.py --split random   --label_col "$LABEL" --splits_dir "$SPLITS_DIR"
python scripts/shift_diagnostics.py --split temporal --label_col "$LABEL" --splits_dir "$SPLITS_DIR"
python scripts/shift_diagnostics.py --split hospital --label_col "$LABEL" --splits_dir "$SPLITS_DIR"
python scripts/make_latex_tables.py

echo "All done. See:"
echo " - results/ (CSV metrics)"
echo " - results/preds/ (per-example preds)"
echo " - results/figs/calibration/ (reliability curves)"
echo " - results/figs/compare/ (grouped + delta charts)"
echo " - results/shift/ (*.csv shift diagnostics)"
echo " - results/tables/ (LaTeX tables to \\input)"
