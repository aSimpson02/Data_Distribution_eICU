#!/usr/bin/env python3
from __future__ import annotations
import sys, glob
from pathlib import Path
import pandas as pd
import numpy as np

# Columns we care about (if missing, we’ll fill with NaN)
CORE_COLS = [
    "scenario","model","seed","calib","feat_select","n_features",
    "auroc","auprc","brier","nll","ece","tau"
]

def load_csvs(patterns):
    frames = []
    for pat in patterns:
        for fp in glob.glob(pat):
            try:
                df = pd.read_csv(fp)
                df["__src__"] = Path(fp).name
                frames.append(df)
            except Exception as e:
                print(f"[WARN] failed to read {fp}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Normalize column names to lower
    out.columns = [c.lower() for c in out.columns]
    # Add any missing expected cols
    for c in CORE_COLS:
        if c not in out.columns:
            out[c] = np.nan
    # Derive scenario/model from filename if absent
    if out["scenario"].isna().all():
        # try to parse from filename pattern like: random_lr_platt_none_seed42.csv
        def parse_scenario(name):
            return name.split("_")[0]
        out["scenario"] = out["__src__"].apply(parse_scenario)
    if out["model"].isna().all():
        def parse_model(name):
            # examples: random_lr_platt_none_seed42.csv | random_xgb_notuned_seed42.csv
            parts = name.split("_")
            return parts[1] if len(parts) > 1 else np.nan
        out["model"] = out["__src__"].apply(parse_model)
    return out

def main():
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results/ directory found.")
        sys.exit(0)

    # Load tuned (val-calibrated) runs and no-tune runs
    tuned = load_csvs([str(results_dir / "*_*_*_*_seed*.csv")])
    notuned = load_csvs([str(results_dir / "*_*_notuned_seed*.csv")])

    if tuned.empty and notuned.empty:
        print("No metrics found.")
        sys.exit(0)

    # Mark sources
    if not tuned.empty:
        tuned["source"] = "tuned"
    if not notuned.empty:
        notuned["source"] = "notuned"

    # Concatenate for optional overview (not strictly needed)
    all_df = pd.concat([df for df in [tuned, notuned] if not df.empty], ignore_index=True)

    # --- Pick best tuned per (scenario, model) by AUROC ---
    if not tuned.empty:
        tuned_best = tuned.sort_values(["scenario","model","auroc"], ascending=[True, True, False]) \
                         .groupby(["scenario","model"], as_index=False).head(1)
        tuned_best = tuned_best[CORE_COLS + ["source","__src__"]]
    else:
        tuned_best = pd.DataFrame(columns=CORE_COLS + ["source","__src__"])

    # --- Pick corresponding no-tune rows (tau=0.5 fixed) ---
    if not notuned.empty:
        # If multiple seeds exist, keep the first per (scenario, model)
        nt_best = notuned.sort_values(["scenario","model","auroc"], ascending=[True, True, False]) \
                         .groupby(["scenario","model"], as_index=False).head(1)
        nt_best = nt_best[CORE_COLS + ["source","__src__"]]
    else:
        nt_best = pd.DataFrame(columns=CORE_COLS + ["source","__src__"])

    # --- Merge for side-by-side comparison ---
    comp = nt_best.merge(
        tuned_best,
        on=["scenario","model"],
        how="outer",
        suffixes=("_no_tune","_tuned")
    )

    # Compute deltas where we have both rows
    comp["delta_auroc"] = comp["auroc_tuned"] - comp["auroc_no_tune"]
    comp["delta_ece"]   = comp["ece_tuned"]   - comp["ece_no_tune"]
    comp["delta_brier"] = comp["brier_tuned"] - comp["brier_no_tune"]
    comp["delta_nll"]   = comp["nll_tuned"]   - comp["nll_no_tune"]

    # Select tidy output columns
    view_cols = [
        "scenario","model",
        "auroc_no_tune","ece_no_tune","brier_no_tune","nll_no_tune",
        "auroc_tuned","ece_tuned","brier_tuned","nll_tuned",
        "delta_auroc","delta_ece","delta_brier","delta_nll",
        "calib_tuned","feat_select_tuned","n_features_tuned",
        "__src___no_tune","__src___tuned"
    ]
    # Rename columns for clarity
    comp = comp.rename(columns={
        "__src___no_tune":"file_no_tune",
        "__src___tuned":"file_tuned",
        "calib_tuned":"calib",
        "feat_select_tuned":"feat_select",
        "n_features_tuned":"n_features"
    })

    # Ensure columns exist even if some side missing
    for c in view_cols:
        if c not in comp.columns:
            comp[c] = np.nan

    comp = comp[["scenario","model",
                 "auroc_no_tune","ece_no_tune","brier_no_tune","nll_no_tune",
                 "auroc_tuned","ece_tuned","brier_tuned","nll_tuned",
                 "delta_auroc","delta_ece","delta_brier","delta_nll",
                 "calib","feat_select","n_features",
                 "file_no_tune","file_tuned"]]

    # Sort for readability
    comp = comp.sort_values(["scenario","model"]).reset_index(drop=True)

    # Print to terminal
    with pd.option_context("display.max_columns", None, "display.width", 160):
        if comp.empty:
            print("No comparable rows found (did you run both no-tune and tuned steps?).")
        else:
            print(comp.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x,float) else str(x)))

    # Save
    outp = results_dir / "compare_no_tune_vs_tuned.csv"
    comp.to_csv(outp, index=False)
    print(f"\nSaved {outp}")

if __name__ == "__main__":
    main()
