#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from scripts.eval_no_tune import main as run_notune   # reuse CLI
from scripts.tune_with_val import main as run_tuned   # reuse CLI

# This wrapper expects you already prepared per-hospital splits under data/csv_splits_loho/Hx_*.csv
# (or a small helper you write to materialize them). It just loops targets and collects CSVs.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hospitals", nargs="+", required=True, help="e.g. H1 H2 H3 H4 H5 H6")
    ap.add_argument("--label_col", default="hospital_mortality")
    ap.add_argument("--models", nargs="+", default=["lr","rf","xgb","lgbm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()

    # naive loop — assumes rerun_all.sh style split creation per target already done
    results = []
    for h in args.hospitals:
        # here ‘scenario’ simply encodes the target hospital (for filenames)
        # you can point eval scripts to a hospital-specific splits_dir if you generate them.
        pass  # left as a scaffold — keeps repo lean; wire to your split-maker if you want full LOHO

    print("[INFO] loho_eval scaffold added. Wire to your hospital split generator if needed.")

if __name__ == "__main__":
    main()
