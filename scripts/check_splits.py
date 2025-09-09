#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

SPLITS = ("random", "temporal", "hospital")
ROOT = "data/csv_splits"
LABEL_COL_CANDIDATES = [
    "hospital_mortality", "label", "y_true", "y", "in_hospital_mortality",
    "hospital_expire_flag", "mortality", "outcome"
]

def find_label_col(df: pd.DataFrame):
    cols = [c.lower() for c in df.columns]
    for cand in LABEL_COL_CANDIDATES:
        if cand in df.columns: return cand
        if cand in cols: return df.columns[cols.index(cand)]
    raise KeyError("No label column found")

def main():
    for split in SPLITS:
        base = os.path.join(ROOT, split)
        tr, va, te = [os.path.join(base, f"{s}.csv") for s in ("train","val","test")]
        print(f"\n[{split.upper()}]")
        if not all(os.path.exists(p) for p in (tr,va,te)):
            print(f"[WARN] missing one of {{train,val,test}} under {base}")
            continue
        dtr, dva, dte = pd.read_csv(tr), pd.read_csv(va), pd.read_csv(te)
        lab = find_label_col(dtr)
        print(f"  train: n={len(dtr):,}  prevalence={dtr[lab].mean():.4f}")
        print(f"  val:   n={len(dva):,}  prevalence={dva[lab].mean():.4f}")
        print(f"  test:  n={len(dte):,}  prevalence={dte[lab].mean():.4f}")

if __name__ == "__main__":
    main()
