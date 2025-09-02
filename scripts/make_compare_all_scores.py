#!/usr/bin/env python3
from pathlib import Path
import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/all_scores.csv", help="input long scores CSV")
    ap.add_argument("--out",    default="results/compare_all_scores.csv", help="output wide CSV")
    args = ap.parse_args()

    INP = Path(args.scores); OUT = Path(args.out)
    if not INP.exists(): raise SystemExit(f"[ERR] {INP} not found.")

    df = pd.read_csv(INP)
    tag = np.where(df.get("source","").astype(str).str.lower()=="notuned",
                   "notuned",
                   df.get("calib_tag", df.get("calib","none")).astype(str).str.lower())
    tag = pd.Series(tag).replace({"": "none"})
    df["tag"]=tag

    idx=["scenario","model","seed"]
    metrics=["auroc","auprc","brier","nll","ece"]
    wide = df.pivot_table(index=idx, columns="tag", values=metrics, aggfunc="first")
    wide.columns=[f"{m}_{t}" for (m,t) in wide.columns]
    wide = wide.reset_index()
    for m in metrics:
        for t in ["none","platt","isotonic","icl"]:
            ct, cb = f"{m}_{t}", f"{m}_notuned"
            if ct in wide.columns and cb in wide.columns:
                wide[f"delta_{m}_{t}_minus_notuned"] = wide[ct]-wide[cb]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT, index=False)
    print(f"[OK] wrote {OUT} (rows={len(wide)})")

if __name__ == "__main__":
    main()
