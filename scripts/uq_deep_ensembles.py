#!/usr/bin/env python3

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import warnings

# Repo root on sys.path (for src/... and scripts/... imports)
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)

from src.feats.selector import FeaturePolicy
from scripts.build_models import build_model



DROP = {
    "patientunitstayid","hospitalid","hospitaldischargeyear",
    "apachescore","predictedhospitalmortality","admissionoffset"
}
STEM = {
    "iid":"random","random":"random",
    "hospital":"hospital","hospital_ood":"hospital",
    "temporal":"temporal","temporal_ood":"temporal",
}

def ece(y, p, bins=20):
    y = np.asarray(y).astype(int); p = np.asarray(p)
    edges = np.linspace(0,1,bins+1); out = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        m = (p>=lo) & (p<(hi if i<bins-1 else hi+1e-12))
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)

def add_calibration(model, X_val, y_val, method: str):
    if method == "none":
        return model
    if method == "platt":
        return CalibratedClassifierCV(model, method="sigmoid", cv="prefit").fit(X_val, y_val)
    if method == "isotonic":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return CalibratedClassifierCV(model, method="isotonic", cv="prefit").fit(X_val, y_val)
    raise ValueError(method)

def choose_features(df: pd.DataFrame, label_col: str):
    keep=[]
    for c in df.columns:
        if c == label_col or c in DROP:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or c.startswith("diagnosis_bucket_"):
            keep.append(c)
    return keep



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=list(STEM.keys()))
    p.add_argument("--label_col", required=True)

    p.add_argument("--model", required=True, choices=["lr","rf","xgb","lgbm"])
    p.add_argument("--members", type=int, default=5,
                   help="Number of ensemble members")
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="Optional explicit seeds (len==members). If omitted, uses 1..members")

    # calibration
    p.add_argument("--calib", choices=["none","platt","isotonic"], default="none")

    # feature-selection knobs (same interface as your pipeline)
    p.add_argument("--feat_select", choices=["none","vif","rfe","vif_rfe"], default="none")
    p.add_argument("--rfe_keep", type=int, default=30)
    p.add_argument("--missing_thresh", type=float, default=0.40)
    p.add_argument("--vif_thresh", type=float, default=10.0)

    # standardize numeric features post-transform (good for MLP/LR; harmless for trees)
    p.add_argument("--standardize", action="store_true")

    # uncertainty rule & abstention
    p.add_argument("--rule", choices=["var","entropy"], default="var",
                   help="var: variance across members; entropy: of ensemble mean")
    p.add_argument("--target_abstain", type=float, default=None,
                   help="Fraction to abstain (e.g., 0.20 → keep 80% least-uncertain)")
    p.add_argument("--unc_tau", type=float, default=None,
                   help="Direct uncertainty threshold; keep u <= unc_tau")

    # sweep for multiple keep rates
    p.add_argument("--sweep", action="store_true",
                   help="Evaluate multiple keep rates (100/90/80/60/40%)")

    # IO
    p.add_argument("--splits_dir", default="data/csv_splits")
    p.add_argument("--results_dir", default="results/preds_ensembles")
    p.add_argument("--summary_dir", default="results")
    return p.parse_args()



def main():
    args = parse_args()
    stem = STEM[args.scenario]
    d = Path(args.splits_dir)

    # Load splits
    tr = pd.read_csv(d / f"{stem}_train.csv")
    va = pd.read_csv(d / f"{stem}_val.csv")
    te = pd.read_csv(d / f"{stem}_test.csv")

    ytr = tr[args.label_col].astype(int).values
    yva = va[args.label_col].astype(int).values
    yte = te[args.label_col].astype(int).values

    # Feature policy (same as your pipeline)
    start = choose_features(tr, args.label_col)
    policy = FeaturePolicy(
        feat_select=args.feat_select,
        missing_thresh=args.missing_thresh,
        vif_thresh=args.vif_thresh,
        rfe_keep=args.rfe_keep
    ).fit(tr[[args.label_col] + start], label_col=args.label_col)

    Xtr = policy.transform(tr)
    Xva = policy.transform(va)
    Xte = policy.transform(te)

    scaler = None
    if args.standardize:
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xva = scaler.transform(Xva)
        Xte = scaler.transform(Xte)

    # Seeds
    seeds = args.seeds if args.seeds else list(range(1, args.members+1))

    # Train members
    member_preds = []
    used_seeds = []
    for s in seeds:
        try:
            clf = build_model(args.model, s)
        except RuntimeError as e:
            print(f"[SKIP] {args.model} seed={s}: {e}")
            continue

        # fit
        clf.fit(Xtr, ytr)

        # calibrate on val, prefit
        clf_cal = add_calibration(clf, Xva, yva, args.calib)

        # predict on test
        p = clf_cal.predict_proba(Xte)[:, 1]
        member_preds.append(p)
        used_seeds.append(s)

    if len(member_preds) == 0:
        print("No ensemble members trained; abort.")
        return

    P = np.vstack(member_preds)  # [M, N]
    p_mean = P.mean(axis=0)
    p_var  = P.var(axis=0)
    eps = 1e-12
    entropy = -(p_mean*np.log(p_mean+eps) + (1-p_mean)*np.log(1-p_mean+eps))

    # uncertainty score
    unc = p_var if args.rule == "var" else entropy

    # abstention rules
    def apply_abstention(u, target_abstain=None, unc_tau=None):
        keep_mask = np.ones_like(u, dtype=bool)
        tau_used = None
        if target_abstain is not None:
            # keep (1 - target_abstain) fraction by lowest uncertainty
            keep_frac = max(0.0, min(1.0, 1.0 - float(target_abstain)))
            k = int(np.floor(len(u) * keep_frac))
            if k <= 0:
                keep_mask[:] = False
            elif k >= len(u):
                keep_mask[:] = True
            else:
                thr = np.partition(u, k)[k]
                keep_mask = u <= thr
                tau_used = float(thr)
        elif unc_tau is not None:
            keep_mask = u <= float(unc_tau)
            tau_used = float(unc_tau)
        return keep_mask, tau_used

    def pack_metrics(y, p):
        return dict(
            auroc=roc_auc_score(y, p),
            auprc=average_precision_score(y, p),
            brier=brier_score_loss(y, p),
            nll=log_loss(y, p, labels=[0,1]),
            ece=ece(y, p),
        )

    # Directory prep
    out_pred_dir = Path(args.results_dir); out_pred_dir.mkdir(parents=True, exist_ok=True)
    out_sum_dir  = Path(args.summary_dir); out_sum_dir.mkdir(parents=True, exist_ok=True)

    # Base stem for filenames
    base = f"{args.scenario}_{args.model}_{args.calib}_{args.rule}_M{len(used_seeds)}"
    if args.standardize:
        base += "_std"
    if args.feat_select != "none":
        base += f"_{args.feat_select}"

    # Always compute full-test metrics (no abstention)
    met_all = pack_metrics(yte, p_mean)

    # Optional single abstention setting
    single_keep = None
    single_tau = None
    if (args.target_abstain is not None) or (args.unc_tau is not None):
        keep_mask, tau_used = apply_abstention(unc, args.target_abstain, args.unc_tau)
        single_keep = float(keep_mask.mean())
        single_tau = tau_used
        met_keep = pack_metrics(yte[keep_mask], p_mean[keep_mask]) if keep_mask.any() else {}

        # Save per-patient outputs for this setting
        stemname = base
        if args.target_abstain is not None:
            stemname += f"_keep{int(round(100*(1.0-args.target_abstain)))}"
        if tau_used is not None:
            stemname += f"_tau{tau_used:.4f}"

        df_out = pd.DataFrame({
            "idx": np.arange(len(p_mean)),
            "y": yte,
            "p_mean": p_mean,
            "p_var": p_var,
            "entropy": entropy,
            "uncertainty": unc,
            "keep": keep_mask.astype(int),
        })
        df_out.to_csv(out_pred_dir / f"{stemname}.csv", index=False)

        # Append summary row
        summary = dict(
            scenario=args.scenario, model=args.model, members=len(used_seeds),
            calib=args.calib, rule=args.rule, standardize=int(args.standardize),
            feat_select=args.feat_select, rfe_keep=(args.rfe_keep if args.feat_select in ("rfe","vif_rfe") else None),
            missing_thresh=args.missing_thresh, vif_thresh=(args.vif_thresh if args.feat_select in ("vif","vif_rfe") else None),
            keep_rate=single_keep, unc_tau=tau_used, n_features=len(policy.selected_features_),
            **{f"all_{k}": v for k, v in met_all.items()}
        )
        summary.update({f"keep_{k}": v for k, v in met_keep.items()})
        sum_path = out_sum_dir / f"uq_summary_{args.scenario}.csv"
        if sum_path.exists():
            pd.concat([pd.read_csv(sum_path), pd.DataFrame([summary])], ignore_index=True).to_csv(sum_path, index=False)
        else:
            pd.DataFrame([summary]).to_csv(sum_path, index=False)

        print(f"[UQ] {args.scenario} | {args.model} | M={len(used_seeds)} | rule={args.rule} "
              f"| keep={single_keep:.2f} → AUROC(all)={met_all['auroc']:.4f}"
              + (f", AUROC(keep)={met_keep['auroc']:.4f}" if met_keep else ""))

    # optional sweep over keep rates for plotting
    if args.sweep:
        sweep_fracs = [1.0, 0.9, 0.8, 0.6, 0.4]  # kept fractions
        rows = []
        for kf in sweep_fracs:
            target_abstain = 1.0 - kf
            keep_mask, tau_used = apply_abstention(unc, target_abstain, None)
            met_keep = pack_metrics(yte[keep_mask], p_mean[keep_mask]) if keep_mask.any() else {}
            rows.append({
                "keep_frac": kf,
                "unc_tau": tau_used,
                **{f"keep_{k}": v for k, v in met_keep.items()}
            })
        df_sweep = pd.DataFrame(rows)
        stemname = base + "_sweep"
        df_sweep.to_csv(out_pred_dir / f"{stemname}.csv", index=False)
        print(f"[UQ] sweep saved → {out_pred_dir / (stemname + '.csv')}")

    
    df_all = pd.DataFrame({
        "idx": np.arange(len(p_mean)),
        "y": yte,
        "p_mean": p_mean,
        "p_var": p_var,
        "entropy": entropy,
        "uncertainty": unc
    })
    df_all.to_csv(out_pred_dir / f"{base}_all.csv", index=False)
    print(f"[UQ] full predictions saved → {out_pred_dir / (base + '_all.csv')}")


if __name__ == "__main__":
    main()
