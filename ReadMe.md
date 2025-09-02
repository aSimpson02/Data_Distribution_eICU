Robustness Under Distribution Shift in Clinical Tabular Data

Goal: evaluate how standard tabular models (LR/RF/XGB/LGBM) and lightweight in-context learning (ICL) baselines behave under distribution shift on eICU, with post-hoc calibration and no retraining in the target domain.

Shifts tested: random (i.i.d.), temporal, hospital (held-out institution)

Metrics: AUROC, AUPRC, ECE, Brier, Test NLL

Outputs: reproducible CSVs + plots; LaTeX tables ready to \input{...}

If only one section is needed, see Quickstart.

TL;DR — Quickstart
# 0) Python 3.11 in a venv
pip install -r requirements.txt

# 1) Baseline (LR/RF/XGB/LGBM): train → calibrate → compare → plots → LaTeX tables
chmod +x run_eval_and_plots.sh
./run_eval_and_plots.sh

# 2) ICL baselines (TabPFN + Prototype + Prototype++), results under results/icl/*
pip install "torch==2.*" --index-url https://download.pytorch.org/whl/cpu && pip install tabpfn
chmod +x run_icl.sh
./run_icl.sh


Outputs of interest

CSV metrics: results/*.csv, results/icl/*.csv

Reliability curves: results/figs/calibration/*, results/figs/calibration_icl/*

Grouped/delta plots: results/figs/compare/*, results/figs/compare_icl/*

Shift diagnostics: results/shift/*

LaTeX tables: results/tables/*.tex (ready to \input)

Problem statement (short)

Clinical ML models are typically trained under i.i.d. assumptions but deployed under shift (institutions, time, guidelines). This repo benchmarks:

Discrimination (AUROC/AUPRC)

Calibration (ECE, Brier, NLL)

Robustness across random/temporal/hospital splits

using simple, reproducible post-hoc methods (Platt scaling) and ICL baselines that adapt at inference (no retraining).
Label: hospital_mortality (binary).

Data & splits

Place CSVs here (flat or nested; .csv or .csv.gz are supported):

data/csv_splits/
  random_train.csv        (or random/train.csv[.gz])
  random_val.csv
  random_test.csv
  temporal_train.csv
  temporal_val.csv
  temporal_test.csv
  hospital_train.csv
  hospital_val.csv
  hospital_test.csv


The loader will:

accept flat/nested, csv/gz

drop known leaky columns (IDs/severity scores)

sanitize whitespace in column names

YAIB cohorts can be converted; see YAIB integration below.

Repo structure
src/
  data/io.py                 # robust split loader (flat/nested, csv/gz) + leak-column drop
  feats/selector.py          # feature policy (missingness handling, etc.)
scripts/
  eval_no_tune.py            # LR/RF/XGB/LGBM → TEST (no calibration)
  tune_with_val.py           # calibrate on VAL (none/platt), freeze → TEST
  compare_results.py         # tuned vs notuned; writes all_scores.csv, compare_*.csv
  make_calibration_plots.py  # reliability curves (supports custom preds_dir/outdir)
  make_compare_all_scores.py # wide metrics across calibrators (arg-friendly)
  plot_compare_all_scores.py # grouped/delta bar charts (arg-friendly)
  shift_diagnostics.py       # KS/KL/χ² + prevalence deltas; results/shift/*
  make_latex_tables.py       # cohort sizes + AUROC/ECE → results/tables/*.tex
  yaib_export_splits.py      # YAIB parquet → this repo CSV split format
  icl_tabpfn.py              # TabPFN ICL baseline (CPU/GPU)
  icl_prototype.py           # simple k-NN ICL (no training)
  icl_prototype_plus.py      # stronger ICL (PCA + feat weighting + softmax dist + abstention)
run_eval_and_plots.sh        # one-button baseline pipeline
run_icl.sh                   # one-button ICL pipeline

Setup

Python: 3.11 recommended.

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows Git Bash
source .venv/Scripts/activate

pip install --upgrade pip wheel
pip install -r requirements.txt


Windows tip: run from Git Bash (not PowerShell). UTF-8 is already enforced in the .sh files.

TabPFN (ICL):

CPU:
pip install "torch==2.*" --index-url https://download.pytorch.org/whl/cpu && pip install tabpfn

GPU (optional): install the CUDA-matched PyTorch wheel, then pip install tabpfn.

How to run
Baseline pipeline (LR/RF/XGB/LGBM)
chmod +x run_eval_and_plots.sh
./run_eval_and_plots.sh


This performs:

eval_no_tune.py (all models, all splits)

tune_with_val.py with calibrators {none, platt}

consolidation with compare_results.py + make_compare_all_scores.py

reliability curves → results/figs/calibration/…

grouped/delta score plots → results/figs/compare/…

shift diagnostics (KS/KL/χ²/prevalence) + LaTeX tables (cohorts, AUROC/ECE)

ICL pipeline (TabPFN + Prototype + Prototype++)
pip install "torch==2.*" --index-url https://download.pytorch.org/whl/cpu && pip install tabpfn
chmod +x run_icl.sh
./run_icl.sh


Writes everything to results/icl/* and plots to results/figs/calibration_icl/*, results/figs/compare_icl/*.

What the files mean

results/all_scores.csv — long table of every (split × model × calibrator × seed)

results/compare_no_tune_vs_tuned.csv — best tuned vs best notuned per split+model, with deltas

results/compare_all_scores.csv — wide table: AUROC/ECE/Brier/NLL/AUPRC per calibrator tag, with deltas vs notuned

results/preds/*.csv — per-example predictions (y_true, p) for reliability curves

results/figs/calibration/* — reliability diagrams

results/figs/compare/* — grouped bars and delta bars

results/shift/* — per-feature KS/KL/χ² + split summary (mean KS/KL, label prevalence deltas)

results/tables/*.tex — LaTeX tables ready to \input

ICL mirrors the same layout under results/icl/* and its plot folders.

Models & calibration

Models: LR, RF, XGB, LGBM (plus ICL models in scripts/icl_*)

Calibrators: none, Platt

In practice: LR benefits substantially from Platt; boosted trees are often already well-calibrated and Platt can worsen ECE.

ICL baselines

TabPFN (scripts/icl_tabpfn.py): transformer trained on synthetic tabular tasks; treat TRAIN as context; single forward pass per batch. Optional Platt; train cap for memory.

Prototype (scripts/icl_prototype.py): k-NN over standardized features; probability = distance-weighted neighbor label average. Optional Platt.

Prototype++ (scripts/icl_prototype_plus.py): stronger non-parametric ICL:

PCA to 64 dims

feature weights from L1-logistic (or MI)

softmax distance weighting (--weight softmax --temperature 0.1)

optional abstention via margin

optional Platt

Tuning tips

Start with: --metric cosine --weight softmax --temperature 0.05..0.2

For speed: lower --cap_train to 20–30k

For coverage analysis: add --abstain_margin 0.05

YAIB integration (optional)

If YAIB cohorts were generated (parquet), convert to this split format:

python scripts/yaib_export_splits.py \
  --yaib_dir /path/to/yaib/output \
  --stem hospital \
  --label_out hospital_mortality \
  --outdir data/csv_splits


Then re-run either pipeline.

LaTeX hooks (for the thesis)

After run_eval_and_plots.sh, include tables:

% Cohort sizes
\input{results/tables/cohort_sizes.tex}

% Best AUROC/ECE per split+model (best calibrator observed)
\input{results/tables/auroc_ece_best.tex}


Example figures:

\includegraphics[width=0.32\linewidth]{results/figs/calibration/random/lgbm/calibration_random_lgbm_none.png}
\includegraphics[width=0.32\linewidth]{results/figs/calibration/hospital/lr/calibration_hospital_lr_platt.png}
\includegraphics[width=0.32\linewidth]{results/figs/calibration/temporal/xgb/calibration_temporal_xgb_none.png}

Reproducibility

Seed: 42 across scripts

Calibration fit on VAL, frozen before TEST

Transformers/encoders/imputation fit on TRAIN only (FeaturePolicy)

No test leakage: the loader and leakage guard drop known leak variables

Troubleshooting

UTF-8 / console glyphs: UTF-8 is enforced in the .sh files. On Windows, use Git Bash.

TabPFN OOM: lower --cap_train, increase --batch, or run CPU; avoid isotonic.

“No preds found for plotting”: check results/preds/ (or results/icl/preds/) contains *_seed*.csv with y_true,p.

Loader can’t find files: accepted patterns are random_train.csv, random_train.csv.gz, random/train.csv(.gz) (same for {val,test}, {temporal,hospital}).

Extending

New model → add in scripts/build_models.py, reference it in the runners.

New split → drop CSVs into data/csv_splits/ using the same names.

New ICL idea → copy icl_prototype_plus.py, swap the representation/distance, keep the interface (write per-example preds + one-row metrics).

Safety & data notes

Assumes de-identified eICU v2.0; don’t commit PHI or raw hospital identifiers.

.gitignore should exclude data/, results/, artifacts/, .venv/.

License / citation

Add your license and citation lines here. Respect upstream licenses for TabPFN and YAIB.

Handy one-liners

Baseline (everything)

./run_eval_and_plots.sh


ICL only

./run_icl.sh


Shift diagnostics (single split)

python scripts/shift_diagnostics.py --split hospital --label_col hospital_mortality


Reliability curves for ICL (custom dirs)

python scripts/make_calibration_plots.py \
  --splits random temporal hospital \
  --models tabpfn protoicl protoicl_plus \
  --tags none platt \
  --preds_dir results/icl/preds \
  --outdir   results/figs/calibration_icl


Wide compare table for ICL

python scripts/make_compare_all_scores.py \
  --scores results/icl/all_scores.csv \
  --out    results/icl/compare_all_scores.csv


Grouped/delta plots for ICL

python scripts/plot_compare_all_scores.py \
  --in results/icl/compare_all_scores.csv \
  --outdir results/figs/compare_icl \
  --metrics auroc ece brier nll auprc \
  --model_order tabpfn protoicl protoicl_plus