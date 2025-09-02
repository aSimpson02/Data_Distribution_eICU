#!/usr/bin/env python3
from pathlib import Path
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def _available_tags(df: pd.DataFrame, metric: str, tags_pref):
    return [t for t in tags_pref if f"{metric}_{t}" in df.columns]

def _available_delta_tags(df: pd.DataFrame, metric: str):
    out=[]
    for t in ["none","platt","isotonic","icl"]:
        if f"delta_{metric}_{t}_minus_notuned" in df.columns:
            out.append(t)
    return out

def _select_rows(df: pd.DataFrame, scenario: str):
    d = df[df["scenario"] == scenario].copy()
    rk_col = "auroc_notuned"
    if rk_col not in d.columns:
        cand = [c for c in d.columns if c.startswith("auroc_") and not c.startswith("delta_")]
        rk_col = cand[0] if cand else None
    if rk_col:
        d = d.sort_values(["model", rk_col], ascending=[True, False]).groupby("model", as_index=False).head(1)
    return d

def plot_metric_grouped(df, outdir, scenario, metric, tags, model_order):
    d = _select_rows(df, scenario)
    models = [m for m in model_order if m in d["model"].unique().tolist()]
    if not models or not tags: return
    vals=[]
    for m in models:
        row = d[d["model"] == m].head(1)
        vals.append([float(row[f"{metric}_{t}"].values[0]) if f"{metric}_{t}" in row.columns and pd.notna(row[f"{metric}_{t}"].values[0]) else np.nan for t in tags])
    V=np.array(vals); x=np.arange(len(models)); width = 0.8/max(len(tags),1)
    plt.figure()
    for j,t in enumerate(tags): plt.bar(x + j*width, V[:,j], width, label=t)
    plt.xticks(x + (len(tags)-1)*width/2, models); plt.ylabel(metric.upper())
    plt.title(f"{scenario} — {metric.upper()} by model/calibration")
    plt.legend(); plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{metric}_{scenario}_grouped.png"
    plt.savefig(out, dpi=160); plt.close(); print(f"[OK] {out}")

def plot_metric_deltas(df, outdir, scenario, metric, delta_tags, model_order):
    d = _select_rows(df, scenario)
    models = [m for m in model_order if m in d["model"].unique().tolist()]
    if not models or not delta_tags: return
    for t in delta_tags:
        col=f"delta_{metric}_{t}_minus_notuned"
        y=[float(d[d['model']==m][col].values[0]) if col in d.columns and not d[d['model']==m].empty and pd.notna(d[d['model']==m][col].values[0]) else np.nan for m in models]
        x=np.arange(len(models))
        plt.figure(); plt.bar(x, y); plt.xticks(x, models); plt.axhline(0, linestyle="--")
        plt.ylabel(f"Δ{metric.upper()} vs notuned ({t})"); plt.title(f"{scenario} — Δ{metric.upper()} ({t}−notuned)")
        plt.tight_layout()
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"delta_{metric}_{t}_{scenario}.png"
        plt.savefig(out, dpi=160); plt.close(); print(f"[OK] {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/compare_all_scores.csv")
    ap.add_argument("--outdir", default="results/figs/compare")
    ap.add_argument("--metrics", nargs="*", default=["auroc","ece"])
    ap.add_argument("--model_order", nargs="*", default=["lr","rf","xgb","lgbm","tabpfn","protoicl"])
    ap.add_argument("--tags_pref", nargs="*", default=["notuned","none","platt","isotonic","icl"])
    args = ap.parse_args()

    INP = Path(args.inp); OUTD = Path(args.outdir)
    if not INP.exists(): raise SystemExit(f"[ERR] {INP} not found.")
    df = pd.read_csv(INP)
    if "scenario" not in df.columns or "model" not in df.columns:
        raise SystemExit("[ERR] compare_all_scores missing 'scenario'/'model'.")

    scenarios = sorted(df["scenario"].unique().tolist())
    for sc in scenarios:
        for metric in args.metrics:
            tags = _available_tags(df, metric, args.tags_pref)
            if tags:       plot_metric_grouped(df, OUTD, sc, metric, tags, args.model_order)
            delta_tags = _available_delta_tags(df, metric)
            if delta_tags: plot_metric_deltas(df, OUTD, sc, metric, delta_tags, args.model_order)

if __name__ == "__main__":
    main()
