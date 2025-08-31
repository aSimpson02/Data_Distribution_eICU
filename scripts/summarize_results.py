#!/usr/bin/env python3
from __future__ import annotations
import sys, re
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")

def load_all(scenario: str|None=None, model: str|None=None) -> pd.DataFrame:
    rows = []
    for p in RESULTS_DIR.glob("*.csv"):
        m = re.match(r"^(random|temporal|hospital)_(lr|rf)_(none|platt|isotonic)_(\w+)_seed(\d+)\.csv$", p.name)
        if not m: 
            continue
        scen, mdl, calib, feat, seed = m.groups()
        if scenario and scen != scenario: 
            continue
        if model and mdl != model: 
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.shape[0] >= 1:
            r = df.iloc[0].to_dict()
            r.update(dict(scenario=scen, model=mdl, calib=calib, feat=feat, seed=int(seed)))
            rows.append(r)
    return pd.DataFrame(rows)

def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    model    = sys.argv[2] if len(sys.argv) > 2 else None
    df = load_all(scenario, model)
    if df.empty:
        print("No metrics found.")
        return
    preferred = ["scenario","model","seed","calib","feat",
                 "auroc","auprc","brier","nll","ece","f1_at_tau","balacc_at_tau","tau","n_features","n_feats"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols].sort_values(["scenario","model","calib","seed"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x,float) else str(x)))
    out = RESULTS_DIR / "results_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    main()
