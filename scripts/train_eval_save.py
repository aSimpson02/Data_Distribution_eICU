#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

# external libs
_HAS_XGB = True
_HAS_LGBM = True
try:
    from xgboost import XGBClassifier
except Exception:
    _HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
except Exception:
    _HAS_LGBM = False

SPLIT_ROOT = "data/csv_splits"
PREDS_DIR = "models/preds"
OUT_METRICS = "results/compare_all_scores.csv"
OUT_TABLE_DIR = "results/tables"
OUT_TABLE_TEX = os.path.join(OUT_TABLE_DIR, "mainresults.tex")

LABEL_CANDS = [
    "hospital_mortality", "label", "y_true", "y",
    "in_hospital_mortality", "hospital_expire_flag", "mortality", "outcome"
]




def find_label_col(df: pd.DataFrame) -> str:
    cols_lower = [c.lower() for c in df.columns]
    for cand in LABEL_CANDS:
        if cand in df.columns:
            return cand
        if cand in cols_lower:
            return df.columns[cols_lower.index(cand)]
    raise KeyError(f"No label column found. Tried {LABEL_CANDS}")


def ece_score(y_true, y_prob, n_bins: int = 15) -> float:
    """Expected Calibration Error (binary)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    inds = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        p_bin = y_prob[mask].mean()
        y_bin = y_true[mask].mean()
        ece += abs(p_bin - y_bin) * mask.mean()
    return float(ece)


def build_preprocessor(df: pd.DataFrame, label_col: str) -> ColumnTransformer:
    feature_cols = [c for c in df.columns if c != label_col]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_tf, num_cols),
            ("cat", cat_tf, cat_cols),
        ]
    )


def get_models():
    models = {
        "lr": LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        "rf": RandomForestClassifier(
            n_estimators=400, n_jobs=-1, class_weight="balanced_subsample", random_state=42
        ),
    }
    if _HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
        )
    if _HAS_LGBM:
        models["lgbm"] = LGBMClassifier(
            n_estimators=800, learning_rate=0.05, num_leaves=64,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
    return models


def save_preds(split, model, calibrator, set_name, y_true, y_prob):
    out = pd.DataFrame({
        "split": split,
        "set": set_name,
        "model": model,
        "calibrator": calibrator,
        "y_true": y_true,
        "y_pred_proba": y_prob
    })
    os.makedirs(PREDS_DIR, exist_ok=True)
    path = os.path.join(PREDS_DIR, f"{split}_{model}_{calibrator}_{set_name}.csv")
    out.to_csv(path, index=False)


def fit_and_eval(split, model_name, base_clf, preproc, Xtr, ytr, Xva, yva, Xte, yte):
    rows = []

    # Raw (uncalibrated)
    pipe = Pipeline([("pre", preproc), ("clf", base_clf)])
    pipe.fit(Xtr, ytr)

    # Save val/test raw preds; compute test metrics
    for set_name, X, y in [("val", Xva, yva), ("test", Xte, yte)]:
        probs = pipe.predict_proba(X)[:, 1]
        save_preds(split, model_name, "none", set_name, y, probs)
        if set_name == "test":
            rows.append(dict(
                split=split, model=model_name, calibrator="none",
                auroc=roc_auc_score(y, probs),
                auprc=average_precision_score(y, probs),
                ece=ece_score(y, probs),
                nll=log_loss(y, probs),
                brier=brier_score_loss(y, probs)
            ))

    # Platt calibration on validation
    try:
        Xva_pre = preproc.transform(Xva)
        Xte_pre = preproc.transform(Xte)
        base = pipe.named_steps["clf"]

        # Backward-compatible constructor: sklearn >=1.2 uses 'estimator'; older uses 'base_estimator'
        try:
            cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv="prefit")
        except TypeError:
            cal = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv="prefit")

        cal.fit(Xva_pre, yva)

        for set_name, Xp, y in [("val", Xva_pre, yva), ("test", Xte_pre, yte)]:
            probs = cal.predict_proba(Xp)[:, 1]
            save_preds(split, model_name, "platt", set_name, y, probs)
            if set_name == "test":
                rows.append(dict(
                    split=split, model=model_name, calibrator="platt",
                    auroc=roc_auc_score(y, probs),
                    auprc=average_precision_score(y, probs),
                    ece=ece_score(y, probs),
                    nll=log_loss(y, probs),
                    brier=brier_score_loss(y, probs)
                ))
    except Exception as e:
        print(f"[WARN] calibration failed for {split}/{model_name}: {e}")

    return rows




def main():
    Path(PREDS_DIR).mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    Path(OUT_TABLE_DIR).mkdir(parents=True, exist_ok=True)

    splits = [s for s in ("random","temporal","hospital")
              if (Path(SPLIT_ROOT) / s / "train.csv").exists()]
    all_rows = []

    for split in splits:
        base = Path(SPLIT_ROOT) / split
        ftr, fva, fte = base/"train.csv", base/"val.csv", base/"test.csv"
        if not (ftr.exists() and fva.exists() and fte.exists()):
            print(f"[WARN] {split}: missing CSVs, skipping.")
            continue

        dtr, dva, dte = pd.read_csv(ftr), pd.read_csv(fva), pd.read_csv(fte)
        label_col = find_label_col(dtr)

        preproc = build_preprocessor(dtr, label_col)
        Xtr, ytr = dtr.drop(columns=[label_col]), dtr[label_col].astype(int).values
        Xva, yva = dva.drop(columns=[label_col]), dva[label_col].astype(int).values
        Xte, yte = dte.drop(columns=[label_col]), dte[label_col].astype(int).values

        for model_name, base_clf in get_models().items():
            try:
                rows = fit_and_eval(split, model_name, base_clf, preproc, Xtr, ytr, Xva, yva, Xte, yte)
                all_rows.extend(rows)
            except Exception as e:
                print(f"[WARN] {split}/{model_name}: {e}")

    if not all_rows:
        print("[FATAL] No results written.")
        return


    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(OUT_METRICS, index=False)
    print(f"[OK] wrote {OUT_METRICS} with {len(metrics_df)} rows")


    # round for display; keep AUROC/ECE only in the tabular
    disp = metrics_df.copy()
    disp = disp[["split","model","calibrator","auroc","ece"]]
    # pretty ordering
    split_order = {"hospital": 0, "random": 1, "temporal": 2}
    model_order = {"lrb": 0, "lr": 1, "rf": 2, "xgb": 3, "lgbm": 4}  # safe default
    disp["split_ord"] = disp["split"].map(split_order).fillna(99)
    disp["model_ord"] = disp["model"].map(model_order).fillna(99)
    disp = disp.sort_values(["split_ord","model_ord","calibrator"])
    disp = disp.drop(columns=["split_ord","model_ord"])
    # title case split names and short cal
    disp["split"] = disp["split"].str.capitalize()
    disp["calibrator"] = disp["calibrator"].replace({"none":"none","platt":"platt"})
    disp = disp.rename(columns={
        "split": "Split",
        "model": "Model",
        "calibrator": "Calibrator",
        "auroc": "AUROC",
        "ece": "ECE"
    })
    # format floats to 4dp for the LaTeX tabular, but keep values numeric in CSV
    disp_fmt = disp.copy()
    for col in ("AUROC","ECE"):
        disp_fmt[col] = disp_fmt[col].map(lambda x: f"{x:.4f}")

    # write tabular-only .tex (no nested \begin{table})
    col_format = "lllll"  # l columns; you may wrap with \resizebox in main.tex
    tex = disp_fmt.to_latex(index=False, escape=False, column_format=col_format)
    with open(OUT_TABLE_TEX, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[OK] wrote {OUT_TABLE_TEX}")


if __name__ == "__main__":
    main()
