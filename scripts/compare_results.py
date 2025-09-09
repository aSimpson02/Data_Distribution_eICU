#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

REQUIRED_METRIC_COLS = {
    "scenario", "model", "seed", "auroc", "auprc", "brier", "nll", "ece"
}


_FN_RE = re.compile(r"^([a-z0-9_]+)_([a-z0-9]+)_(.+?)_seed(\d+)\.csv$", re.I)

def parse_from_filename(name: str) -> dict:
    """
    Accepts e.g.
      random_lr_notuned_seed42.csv
      hospital_xgb_platt_none_seed42.csv
      temporal_lgbm_none_none_seed42.csv
    Returns dict with scenario, model, seed, calib_tag (best-effort).
    """
    m = _FN_RE.match(name)
    out = {"scenario": np.nan, "model": np.nan, "seed": np.nan, "calib_tag": np.nan}
    if not m:
        return out
    scenario, model, mid, seed = m.groups()
    out.update({"scenario": scenario, "model": model, "seed": int(seed)})

    # Derive calibration tag from mid / notuned
    mid_l = mid.lower()
    if "notuned" in mid_l:
        out["calib_tag"] = "notuned"
    else:
        # common encodings: "platt_none", "none_none", "isotonic_none"
        out["calib_tag"] = mid_l.split("_")[0]
    return out



def is_metrics_csv(p: Path) -> bool:
    """Only 1-row metrics in results/ (exclude preds/, figs/, summaries)."""
    if p.parent.name != "results": return False
    if p.suffix.lower() != ".csv": return False
    name = p.name.lower()
    if name in {"results_summary.csv", "compare_no_tune_vs_tuned.csv", "all_scores.csv"}:
        return False
    if name.startswith("preds_"):  # just in case
        return False
    if "seed" not in name:  # summaries without seed → skip
        return False
    return True

def load_metrics_csv(p: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[WARN] read error {p.name}: {e}")
        return None

    # Lowercase columns for consistency
    df.columns = [c.lower() for c in df.columns]

    # Must contain metric columns; otherwise skip
    if not REQUIRED_METRIC_COLS.issubset(set(df.columns)):
        return None

    # Best-effort fill for scenario/model/seed/calib_tag if missing/NaN
    # (prefer values from inside the CSV; fall back to filename)
    fn_info = parse_from_filename(p.name)
    for k in ("scenario", "model", "seed"):
        if k not in df.columns or df[k].isna().all():
            df[k] = fn_info[k]

    # Identify source: notuned vs tuned
    #  - Files with "notuned" in filename are NOTUNED
    #  - Everything else (with calib none/platt/isotonic) is TUNED
    source = "notuned" if "notuned" in p.name.lower() else "tuned"
    df["source"] = source

    # Add calib_tag if we can
    if "calib" in df.columns and pd.notna(df.loc[0, "calib"]):
        df["calib_tag"] = df["calib"].astype(str).str.lower()
    else:
        df["calib_tag"] = fn_info["calib_tag"]

    # Keep useful columns; add file name
    df["file"] = p.name
    return df

def collect_all(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in results_dir.iterdir():
        if not is_metrics_csv(p):
            continue
        df = load_metrics_csv(p)
        if df is not None:
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)

    # Ensure optional columns exist
    for c in ["feat", "feat_select", "n_features", "tau", "htgt"]:
        if c not in out.columns:
            out[c] = np.nan
    # Normalize scenario casing
    out["scenario"] = out["scenario"].astype(str).str.lower()
    out["model"] = out["model"].astype(str).str.lower()
    return out



def best_by(df: pd.DataFrame, group_cols: list[str], metric: str,
            ascending: bool, tie_breaker: tuple[str, bool]) -> pd.DataFrame:
    """
    Pick one row per group maximizing/minimizing 'metric'.
    Tie-breaker is a (col, ascending) pair.
    """
    outs = []
    for _, g in df.groupby(group_cols, dropna=False):
        g = g.copy()
        # main metric
        g["_rk1"] = g[metric].rank(method="min", ascending=ascending)
        g1 = g[g["_rk1"] == g["_rk1"].min()].copy()
        # tie-breaker
        tb_col, tb_asc = tie_breaker
        g1["_rk2"] = g1[tb_col].rank(method="min", ascending=tb_asc)
        outs.append(g1.loc[g1["_rk2"].idxmin()])
    out = pd.DataFrame(outs).drop(columns=[c for c in ["_rk1", "_rk2"] if c in outs[0].index])
    return out.reset_index(drop=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", help="Directory with result CSVs.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print("No results/ directory found.")
        return

    all_scores = collect_all(results_dir)
    if all_scores.empty:
        print("No metrics CSVs found in results/.")
        return

    # Save all scores table (long form)
    out_all = results_dir / "all_scores.csv"
    # Order columns nicely
    cols_order = [
        "scenario","htgt","model","seed","source","calib","calib_tag","feat","feat_select","n_features",
        "auroc","auprc","brier","nll","ece","tau","file"
    ]
    cols_present = [c for c in cols_order if c in all_scores.columns]
    other_cols = [c for c in all_scores.columns if c not in cols_present]
    all_scores[cols_present + other_cols].to_csv(out_all, index=False)
    print(f"[OK] wrote {out_all} ({len(all_scores)} rows)")

    # Build side-by-side compare: best tuned vs best notuned per (scenario, model)
    tuned = all_scores[all_scores["source"] == "tuned"].copy()
    notuned = all_scores[all_scores["source"] == "notuned"].copy()

    # Choose best rows:
    #  - tuned: maximize AUROC (tie-break lower ECE)
    #  - notuned: maximize AUROC (tie-break lower ECE)
    tuned_best = pd.DataFrame()
    if not tuned.empty:
        tuned_best = best_by(tuned, ["scenario","model"], metric="auroc",
                             ascending=False, tie_breaker=("ece", True))
    notuned_best = pd.DataFrame()
    if not notuned.empty:
        notuned_best = best_by(notuned, ["scenario","model"], metric="auroc",
                               ascending=False, tie_breaker=("ece", True))

    # Merge side-by-side
    comp = notuned_best.merge(
        tuned_best,
        on=["scenario","model"],
        how="outer",
        suffixes=("_no_tune","_tuned")
    )

    # Compute deltas (tuned - no_tune)
    for m in ["auroc","ece","brier","nll"]:
        comp[f"delta_{m}"] = comp.get(f"{m}_tuned") - comp.get(f"{m}_no_tune")

    # Select tidy view
    view_cols = [
        "scenario","model",
        "auroc_no_tune","ece_no_tune","brier_no_tune","nll_no_tune",
        "auroc_tuned","ece_tuned","brier_tuned","nll_tuned",
        "delta_auroc","delta_ece","delta_brier","delta_nll",
        "calib_tuned","feat_select_tuned","n_features_tuned",
        "file_no_tune","file_tuned"
    ]
    for c in view_cols:
        if c not in comp.columns:
            comp[c] = np.nan
    comp = comp[view_cols].sort_values(["scenario","model"]).reset_index(drop=True)

    # Save compare CSV
    out_comp = results_dir / "compare_no_tune_vs_tuned.csv"
    comp.to_csv(out_comp, index=False)
    print(f"[OK] wrote {out_comp} ({len(comp)} rows)")

    # Pretty print
    with pd.option_context("display.max_columns", None, "display.width", 160):
        if comp.empty:
            print("No comparable rows found (did you run both no-tune and tuned steps?).")
        else:
            def ff(x):
                try:
                    return f"{x:.4f}"
                except Exception:
                    return str(x)
            print(comp.to_string(index=False, formatters={c: ff for c in comp.columns}))

if __name__ == "__main__":
    main()
