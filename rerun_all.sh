#!/usr/bin/env bash
set -euo pipefail


# ★★★★ Config (edit as needed) ★★★★

SCENARIOS=("random" "temporal" "hospital")
MODELS=("lr" "rf" "xgb" "lgbm")
CALIBS=("none" "platt")   # drop isotonic here to speed up; it's heavy but optional
SEED=42
LABEL="hospital_mortality"

# ★★★★ TabFM options (used by eval_tabfm.py) ★★★★
# leave these commented; we’re disabling TabFM below
# TABFM_DEVICE="cpu"
# TABFM_CAP_TRAIN=4000
# TABFM_FEAT_SELECT="vif_rfe"
# TABFM_VIF=5
# TABFM_RFE_KEEP=64
# TABFM_ENSEMBLES=4
# TABFM_POSTERIOR=4
# TABFM_CALIB="isotonic"

# ★★★★ Switches — turn steps on/off ★★★★
DO_EXPORT_SPLITS=0   # you already flattened CSVs; skip noisy warnings
DO_AUDIT=1
DO_EVAL_NOTUNE=1
DO_TUNE_VAL=1
DO_TABFM=0           # <— disable TabPFN (this was causing the OOM)
DO_UQ_ENSEMBLES=0    # <— deep ensembles are heavy; turn off unless you need them
DO_SUMMARIES=1
DO_PLOTS=0
DO_CONFORMAL=0

# ★★★★ Prep & logging ★★★★

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

ts="$(date +"%Y%m%d-%H%M%S")"
mkdir -p "artifacts/${ts}" results data/csv_splits
[ -d results ] && mv results "artifacts/${ts}/results_old" || true
mkdir -p results

LOG="artifacts/${ts}/run.log"
exec > >(tee -a "$LOG") 2>&1

# Color helpers
bold() { printf "\033[1m%s\033[0m\n" "$*"; }
star() { echo "★★★★ $* ★★★★"; }

# Helper: check if LightGBM is importable; if not, drop it
if ! python - <<'PY' >/dev/null 2>&1
import importlib; importlib.import_module("lightgbm")
PY
then
  bold "LightGBM not available → will skip 'lgbm' runs."
  # filter MODELS to remove lgbm
  NEWMODELS=()
  for m in "${MODELS[@]}"; do
    [[ "$m" == "lgbm" ]] || NEWMODELS+=("$m")
  done
  MODELS=("${NEWMODELS[@]}")
fi

# Helper: join arrays (for printing)
join_by() { local IFS="$1"; shift; echo "$*"; }

bold "Run started: $ts"
bold "Scenarios:  $(join_by , "${SCENARIOS[@]}")"
bold "Models:     $(join_by , "${MODELS[@]}")"
bold "Calibrators: $(join_by , "${CALIBS[@]}")"
bold "Seed:       $SEED"
bold "Label:      $LABEL"
echo


# ★★★★ 1) Export CSV splits ★★★★

if [[ "$DO_EXPORT_SPLITS" == "1" ]]; then
  star "★★★★ 1) Export CSV splits ★★★★"
  python scripts/export_splits_to_csv.py \
    --indir data/splits \
    --outdir data/csv_splits \
    --splits "${SCENARIOS[@]/random/random}"  # uses stems: random temporal hospital
fi


# ★★★★ 2) Audit splits ★★★★

if [[ "$DO_AUDIT" == "1" ]]; then
  star "★★★★ 2) Audit splits ★★★★"
  for S in "${SCENARIOS[@]}"; do
    python scripts/audit_split.py "$S" || true
  done
fi


# ★★★★ 3A) Evaluate (NO TUNE) ★★★★

if [[ "$DO_EVAL_NOTUNE" == "1" ]]; then
  star "★★★★ 3A) Evaluate NO-TUNE (train→test, no calibration, τ=0.5) ★★★★"
  for S in "${SCENARIOS[@]}"; do
    python scripts/eval_no_tune.py \
      --scenario "$S" \
      --models "${MODELS[@]}" \
      --seeds "$SEED" \
      --label_col "$LABEL"
  done
fi


# ★★★★ 3B) Train + Tune + Calibrate ★★★★

if [[ "$DO_TUNE_VAL" == "1" ]]; then
  star "★★★★ 3B) Train + TUNE (calibration on val; thresholds on val) ★★★★"
  for S in "${SCENARIOS[@]}"; do
    python scripts/tune_with_val.py \
      --scenario "$S" \
      --models "${MODELS[@]}" \
      --calib "${CALIBS[@]}" \
      --seeds "$SEED" \
      --label_col "$LABEL"
  done
fi


# ★★★★ 3C) TabFM / TabPFN-style baseline ★★★★

if [[ "$DO_TABFM" == "1" ]]; then
  star "★★★★ 3C) TabFM (compact feature set, small cap for quick runs) ★★★★"
  for S in "${SCENARIOS[@]}"; do
    python scripts/eval_tabfm.py \
      --scenario "$S" \
      --label_col "$LABEL" \
      --calib "$TABFM_CALIB" \
      --seeds "$SEED" \
      --device "$TABFM_DEVICE" \
      --cap_train "$TABFM_CAP_TRAIN" \
      --feat_select "$TABFM_FEAT_SELECT" \
      --vif_thresh "$TABFM_VIF" \
      --rfe_keep "$TABFM_RFE_KEEP" \
      --ensembles "$TABFM_ENSEMBLES" \
      --posterior "$TABFM_POSTERIOR"
  done
fi


# ★★★★ 4) Summaries & compare ★★★★

if [[ "$DO_SUMMARIES" == "1" ]]; then
  star "★★★★ 4) Summaries & comparison ★★★★"
  python scripts/summarize_results.py         # prints and writes results/results_summary.csv
  python scripts/compare_results.py           # writes results/compare_tuned_vs_not.csv
fi


# ★★★★ 5) UQ: Deep Ensembles ★★★★

if [[ "$DO_UQ_ENSEMBLES" == "1" ]]; then
  star "★★★★ 5) UQ: Deep Ensembles (keep 80% by variance + entropy; also save sweep) ★★★★"
  for S in "${SCENARIOS[@]}"; do
    # variance-based, keep 80%
    for M in "${MODELS[@]}"; do
      python scripts/uq_deep_ensembles.py \
        --scenario "$S" --label_col "$LABEL" \
        --model "$M" --members 5 --calib platt \
        --rule var --target_abstain 0.20 \
        --sweep
      # entropy-based, keep 80%
      python scripts/uq_deep_ensembles.py \
        --scenario "$S" --label_col "$LABEL" \
        --model "$M" --members 5 --calib platt \
        --rule entropy --target_abstain 0.20
    done
  done
fi


# ★★★★6) Calibration plots (optional)★★★★

if [[ "$DO_PLOTS" == "1" ]]; then
  star "★★★★ 6) Calibration plots (optional) ★★★★"
  for S in "${SCENARIOS[@]}"; do
    python scripts/make_calibration_plots.py --scenario "$S" || true
  done
fi



# 7) Conformal (optional)

if [[ "$DO_CONFORMAL" == "1" ]]; then
  star "7) Conformal prediction (optional)"
  for S in "${SCENARIOS[@]}"; do
    python scripts/conformal_prediction.py \
      --scenario "$S" \
      --models lr rf \
      --calib isotonic platt \
      --alpha 0.10 \
      --seed "$SEED" \
      --label_col "$LABEL"
  done
fi

bold "All done  (results in ./results, log: $LOG)"
