#!/usr/bin/env bash
set -Eeuo pipefail

# --- Config (override via env) ---
PY="${VENV_PY:-./.venv/Scripts/python.exe}"
SCENARIOS="${SCENARIOS:-random temporal hospital}"
LABEL_COL="${LABEL_COL:-hospital_mortality}"
SPLITS_DIR="${SPLITS_DIR:-data/csv_splits}"
RESULTS_DIR="${RESULTS_DIR:-results}"
FEAT_SELECT="${FEAT_SELECT:-none}"    # none | vif | rfe | vif_rfe
VIF_THRESH="${VIF_THRESH:-10.0}"
RFE_KEEP="${RFE_KEEP:-64}"
RUN_PLOTS="${RUN_PLOTS:-1}"
RUN_SHIFT="${RUN_SHIFT:-1}"
# ----------------------------------

# UTF-8 + make src importable
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 LANG=C.UTF-8 LC_ALL=C.UTF-8
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
[ -f src/__init__.py ] || : > src/__init__.py
[ -f src/feats/__init__.py ] || : > src/feats/__init__.py

say(){ echo -e "\n>>> $*\n"; }
exists(){ [[ -f "$1" ]]; }

say "Python sanity"
"$PY" - <<'PY'
import sys, importlib
print("python:", sys.version)
print("exe    :", sys.executable)
for m in ("numpy","pandas","sklearn","lightgbm"):
    try: importlib.import_module(m); print(m,"OK")
    except Exception as e: print(m,"MISSING:",e)
PY

# 1) No-tune
if exists scripts/eval_no_tune.py; then
  for S in $SCENARIOS; do
    say "eval_no_tune.py --scenario $S"
    "$PY" scripts/eval_no_tune.py \
      --scenario "$S" \
      --label_col "$LABEL_COL" \
      --splits_dir "$SPLITS_DIR" \
      --results_dir "$RESULTS_DIR" \
      --feat_select "$FEAT_SELECT" \
      --vif_thresh "$VIF_THRESH" \
      --rfe_keep "$RFE_KEEP"
  done
else
  echo "SKIP: scripts/eval_no_tune.py not found"
fi

# 2) Calibrated (train→val→test)
if exists scripts/tune_with_val.py; then
  for S in $SCENARIOS; do
    say "tune_with_val.py --scenario $S"
    "$PY" scripts/tune_with_val.py \
      --scenario "$S" \
      --label_col "$LABEL_COL" \
      --splits_dir "$SPLITS_DIR" \
      --results_dir "$RESULTS_DIR" \
      --feat_select "$FEAT_SELECT" \
      --vif_thresh "$VIF_THRESH" \
      --rfe_keep "$RFE_KEEP"
  done
else
  echo "SKIP: scripts/tune_with_val.py not found"
fi

# 3) Consolidate & plots (if present)
if exists scripts/compare_results.py; then
  say "compare_results.py"
  "$PY" scripts/compare_results.py
fi

if [[ "$RUN_PLOTS" == "1" ]]; then
  if exists scripts/make_compare_all_scores.py; then
    say "make_compare_all_scores.py"; "$PY" scripts/make_compare_all_scores.py
  fi
  if exists scripts/plot_compare_all_scores.py; then
    say "plot_compare_all_scores.py"; "$PY" scripts/plot_compare_all_scores.py
  fi
  if exists scripts/make_calibration_plots.py; then
    say "make_calibration_plots.py"; "$PY" scripts/make_calibration_plots.py
  fi
fi

# 4) Shift diagnostics (optional)
if [[ "$RUN_SHIFT" == "1" ]] && exists scripts/shift_diagnostics.py; then
  say "shift_diagnostics.py"
  "$PY" scripts/shift_diagnostics.py || echo "shift_diagnostics.py returned non-zero (continuing)"
fi

say "DONE. Outputs in $RESULTS_DIR/"
