#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.data.io import load_three

OUTDIR = Path("results/tables"); OUTDIR.mkdir(parents=True, exist_ok=True)

def cohort_sizes(label="hospital_mortality", splits_dir="data/csv_splits"):
    rows=[]
    for stem in ["random","temporal","hospital"]:
        tr, va, te = load_three(stem, splits_dir)
        rows.append({
            "Split": stem,
            "Train $n$": len(tr),
            "Val $n$": len(va),
            "Test $n$": len(te),
            "Test prevalence": f"{te[label].mean()*100:.2f}\\%"
        })
    df = pd.DataFrame(rows)
    with open(OUTDIR/"cohort_sizes.tex","w") as f:
        f.write(df.to_latex(index=False, escape=False))
    print("[OK] results/tables/cohort_sizes.tex")

def auroc_ece_best(comp_path="results/compare_no_tune_vs_tuned.csv"):
    df = pd.read_csv(comp_path)
    rows=[]
    for (sc, m), g in df.groupby(["scenario","model"]):
        cand=[]
        if "auroc_tuned" in g and pd.notna(g["auroc_tuned"]).any():
            cand.append(("tuned", float(g["auroc_tuned"].values[0]),
                         float(g["ece_tuned"].values[0]),
                         str(g["calib_tuned"].values[0])))
        if "auroc_no_tune" in g and pd.notna(g["auroc_no_tune"]).any():
            cand.append(("none", float(g["auroc_no_tune"].values[0]),
                         float(g["ece_no_tune"].values[0]), "none"))
        if not cand: 
            continue
        cand = sorted(cand, key=lambda x: (x[2], -x[1]))  # ECE asc, AUROC desc
        src, au, ec, calib = cand[0]
        rows.append({"Split": sc, "Model": m, "Calibrator": calib, "AUROC": f"{au:.4f}", "ECE": f"{ec:.4f}"})
    out = pd.DataFrame(rows).sort_values(["Split","Model"])
    with open(OUTDIR/"auroc_ece_best.tex","w") as f:
        f.write(out.to_latex(index=False))
    print("[OK] results/tables/auroc_ece_best.tex")

if __name__ == "__main__":
    cohort_sizes()
    auroc_ece_best()
