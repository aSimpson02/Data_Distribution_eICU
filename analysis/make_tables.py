#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import pandas as pd

def to_tex(df, outpath, caption, label, index=False, float_format="%.4f", max_rows=None):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    if max_rows: df = df.head(max_rows)
    with open(outpath, "w") as f:
        f.write(df.to_latex(index=index, escape=True, longtable=False,
                            caption=caption, label=label, float_format=float_format,
                            bold_rows=False, column_format="l" + "r"*(len(df.columns)-1)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift_dir", default="results/shift")
    ap.add_argument("--perf_link", default="results/shift/perf_link.csv")
    ap.add_argument("--outdir", default="results/tables")
    args = ap.parse_args()

    # Shift signatures (all splits found)
    sig_rows = []
    for split in os.listdir(args.shift_dir):
        sigp = os.path.join(args.shift_dir, split, f"{split}_shift_signature.csv")
        if os.path.exists(sigp):
            df = pd.read_csv(sigp)
            df.insert(0, "split", split)
            sig_rows.append(df)
    if sig_rows:
        sig = pd.concat(sig_rows, ignore_index=True)
        to_tex(sig, os.path.join(args.outdir, "shift_signature.tex"),
               caption="Shift signatures across splits.",
               label="tab:shift-signature", index=False)

    # Top drifters & missingness per split
    for split in os.listdir(args.shift_dir):
        top = os.path.join(args.shift_dir, split, f"{split}_topdrifting.csv")
        mis = os.path.join(args.shift_dir, split, f"{split}_missingness_drift.csv")
        if os.path.exists(top):
            df = pd.read_csv(top).head(15)
            to_tex(df, os.path.join(args.outdir, f"topdrifting_{split}.tex"),
                   caption=f"Top drifting features (by JS) on {split} split.",
                   label=f"tab:topdrifting-{split}", index=False)
        if os.path.exists(mis):
            df = pd.read_csv(mis).head(15)
            to_tex(df, os.path.join(args.outdir, f"missingness_{split}.tex"),
                   caption=f"Features with largest missingness drift on {split} split.",
                   label=f"tab:missingness-{split}", index=False)

    # Perf link table
    if os.path.exists(args.perf_link):
        df = pd.read_csv(args.perf_link)
        cols = [c for c in df.columns if c in
                ("split","model","calibrator","mean_js","mean_ks",
                 "delta_auroc_random","delta_ece_random",
                 "delta_auroc_notuned","delta_ece_notuned")]
        to_tex(df[cols], os.path.join(args.outdir, "perf_link.tex"),
               caption="Shift magnitude and performance deltas by split/model/calibrator.",
               label="tab:perf-link", index=False)

if __name__ == "__main__":
    main()
