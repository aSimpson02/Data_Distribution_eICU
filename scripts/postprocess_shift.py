# scripts/postprocess_shift.py
from __future__ import annotations
import argparse, numpy as np, pandas as pd
from pathlib import Path

def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, int); ranks[order] = np.arange(1, n+1)
    q = p * n / ranks
    q[order[::-1]] = np.minimum.accumulate(q[order[::-1]])
    return np.clip(q, 0, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="CSV with per-feature KS p-values as column 'p_ks'")
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.infile)
    if "p_ks" not in df:
        raise SystemExit("Expected a column 'p_ks' with KS p-values.")
    df["q_ks"] = bh_fdr(df["p_ks"].values)
    df.to_csv(args.outfile, index=False)
    print(f"[OK] wrote {args.outfile}")

if __name__ == "__main__":
    main()
