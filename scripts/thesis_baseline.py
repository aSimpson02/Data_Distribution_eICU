# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse, json, math
# from pathlib import Path
# from typing import List, Tuple, Dict, Optional

# import numpy as np
# import pandas as pd
# from sklearn.metrics import roc_auc_score, brier_score_loss
# from sklearn.linear_model import LogisticRegression
# from sklearn.isotonic import IsotonicRegression
# from lightgbm import LGBMClassifier, early_stopping, log_evaluation
# import matplotlib.pyplot as plt


# DEFAULT_ROOT = Path("yaib_data/mortality24/eicu")
# DEFAULT_SPLITS = ("random", "temporal", "hospital")
# DEFAULT_LABEL_CANDIDATES = ("label", "hospital_mortality")
# DEFAULT_STATIC_BLOCK = {"hospital_mortality", "apachescore", "predictedhospitalmortality"}
# DEFAULT_ECE_BINS = 15
# OUT_ROOT = Path("yaib_logs")




# def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
#     y_true = np.asarray(y_true, dtype=float)
#     y_prob = np.asarray(y_prob, dtype=float)
#     bins = np.linspace(0.0, 1.0, n_bins + 1)
#     idx = np.digitize(y_prob, bins) - 1
#     ece = 0.0
#     n = len(y_true)
#     for b in range(n_bins):
#         m = idx == b
#         if not np.any(m):
#             continue
#         acc = y_true[m].mean()
#         conf = y_prob[m].mean()
#         ece += (m.sum() / n) * abs(acc - conf)
#     return float(ece)


# def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[np.ndarray, np.ndarray]:
#     y_true = np.asarray(y_true, dtype=float)
#     y_prob = np.asarray(y_prob, dtype=float)
#     bins = np.linspace(0.0, 1.0, n_bins + 1)
#     idx = np.digitize(y_prob, bins) - 1
#     bin_conf, bin_acc = [], []
#     for b in range(n_bins):
#         m = idx
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import matplotlib.pyplot as plt



DEFAULT_ROOT = Path("yaib_data/mortality24/eicu")
DEFAULT_SPLITS = ("random", "temporal", "hospital")
DEFAULT_LABEL_CANDIDATES = ("label", "hospital_mortality")
DEFAULT_STATIC_BLOCK = {"hospital_mortality", "apachescore", "predictedhospitalmortality"}
DEFAULT_ECE_BINS = 15
OUT_ROOT = Path("yaib_logs")




def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    bin_conf, bin_acc = [], []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        bin_conf.append(y_prob[m].mean())
        bin_acc.append(y_true[m].mean())
    return np.array(bin_conf), np.array(bin_acc)


def plot_reliability(y_true: np.ndarray, y_prob: np.ndarray, title: str, out_png: Path, n_bins: int = 15):
    ece = ece_score(y_true, y_prob, n_bins=n_bins)
    conf, acc = reliability_curve(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(5.0, 5.0), dpi=120)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Perfect")
    if len(conf):
        order = np.argsort(conf)
        plt.plot(conf[order], acc[order], marker="o", linewidth=1.5, label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical frequency (positive rate)")
    plt.title(f"{title}\nECE={ece:.4f}")
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def load_parquet_split(root: Path, split: str, label_candidates: Tuple[str, ...]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    d = root / split
    tr = pd.read_parquet(d / "train.parquet")
    va = pd.read_parquet(d / "val.parquet")
    te = pd.read_parquet(d / "test.parquet")
    lab = next((c for c in label_candidates if c in tr.columns), None)
    if not lab:
        raise RuntimeError(f"No label column found in {d} (looked for {label_candidates})")
    return tr, va, te, lab


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.replace(r"\s+", "_", regex=True)
    return df


def prepare_train(
    df: pd.DataFrame, lab: str, static_block: set, auc_cut: float = 0.995, eq_cut: float = 0.999
) -> Tuple[pd.DataFrame, np.ndarray, List[str], pd.Series, List[str]]:
    df = normalize_columns(df)
    num = df.select_dtypes(include=[np.number]).copy()
    drop = {lab} | static_block | {c for c in num.columns if "id" in c.lower()}
    X0 = num.drop(columns=[c for c in drop if c in num.columns], errors="ignore")
    y = df[lab].astype(int).to_numpy()

    # dynamic leak scan on TRAIN only
    to_drop = []
    for c in X0.columns:
        x = X0[c].to_numpy()
        try:
            eq = float((x == y).mean()) if np.issubdtype(x.dtype, np.number) else 0.0
        except Exception:
            eq = 0.0
        auc = None
        try:
            if np.unique(x).size > 1:
                auc = float(roc_auc_score(y, x))
        except Exception:
            pass
        if eq >= eq_cut or (auc is not None and auc >= auc_cut):
            to_drop.append(c)

    keep_cols = [c for c in X0.columns if c not in to_drop]
    med = X0[keep_cols].median(numeric_only=True)
    X = X0[keep_cols].fillna(med)
    return X, y, keep_cols, med, to_drop


def prepare_apply(df: pd.DataFrame, lab: str, keep_cols: List[str], med: pd.Series, static_block: set) -> Tuple[pd.DataFrame, np.ndarray]:
    df = normalize_columns(df)
    num = df.select_dtypes(include=[np.number]).copy()
    drop = {lab} | static_block | {c for c in num.columns if "id" in c.lower()}
    X0 = num.drop(columns=[c for c in drop if c in num.columns], errors="ignore")
    X = X0.reindex(columns=keep_cols).fillna(med)
    y = df[lab].astype(int).to_numpy()
    return X, y


def train_lgbm(Xtr, ytr, Xva, yva, seed: int = 42) -> Tuple[LGBMClassifier, Optional[int]]:
    spw = float(max(1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum())))
    clf = LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=spw,
    )
    clf.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        eval_metric="auc",
        callbacks=[early_stopping(100, first_metric_only=True), log_evaluation(0)],
    )
    best = getattr(clf, "best_iteration_", None)
    return clf, (int(best) if best else None)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = DEFAULT_ECE_BINS) -> Dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "ece": float(ece_score(y_true, y_prob, n_bins=n_bins)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def run_one_split(
    root: Path,
    split: str,
    out_root: Path,
    label_candidates: Tuple[str, ...],
    static_block: set,
    ece_bins: int,
    seed: int,
) -> Dict:
    tr, va, te, lab = load_parquet_split(root, split, label_candidates)

    Xtr, ytr, keep, med, auto_drop = prepare_train(tr, lab, static_block)
    Xva, yva = prepare_apply(va, lab, keep, med, static_block)
    Xte, yte = prepare_apply(te, lab, keep, med, static_block)

    model, best_iter = train_lgbm(Xtr, ytr, Xva, yva, seed=seed)
    proba_va = model.predict_proba(Xva, num_iteration=best_iter)[:, 1] if best_iter else model.predict_proba(Xva)[:, 1]
    proba_te = model.predict_proba(Xte, num_iteration=best_iter)[:, 1] if best_iter else model.predict_proba(Xte)[:, 1]

    # Calibration
    platt = LogisticRegression(solver="lbfgs", max_iter=1000).fit(proba_va.reshape(-1, 1), yva)
    te_platt = platt.predict_proba(proba_te.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip").fit(proba_va, yva)
    te_iso = iso.transform(proba_te)

    # Metrics
    base = evaluate(yte, proba_te, n_bins=ece_bins)
    pl_m = evaluate(yte, te_platt, n_bins=ece_bins)
    iso_m = evaluate(yte, te_iso, n_bins=ece_bins)

    # Outputs
    run_dir = out_root / f"eicu_{split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics = {
        "split": split,
        "features_n": len(keep),
        "best_n_estimators": int(best_iter or model.n_estimators),
        "label": lab,
        "base": base,
        "platt": pl_m,
        "isotonic": iso_m,
        "dropped_static": sorted(list(static_block & set(tr.columns))),
        "dropped_auto": auto_drop,
        "keep_cols": keep,
    }
    (run_dir / "calibration.json").write_text(json.dumps(metrics, indent=2))
    (meta_dir / f"keepcols_{split}.json").write_text(json.dumps(keep, indent=2))

    # Save reliability plots
    plot_reliability(yte, proba_te, f"{split.upper()} — Base", run_dir / "reliability_base.png", n_bins=ece_bins)
    plot_reliability(yte, te_platt, f"{split.upper()} — Platt", run_dir / "reliability_platt.png", n_bins=ece_bins)
    plot_reliability(yte, te_iso, f"{split.upper()} — Isotonic", run_dir / "reliability_isotonic.png", n_bins=ece_bins)

    # Console line
    print(
        f"{split.upper():9s}"
        f"  BASE AUC={base['auroc']:.4f} ECE={base['ece']:.4f} Brier={base['brier']:.4f}"
        f" | PLATT ECE={pl_m['ece']:.4f}"
        f" | ISO ECE={iso_m['ece']:.4f}"
        f" | n_feat={len(keep)} | dropped_auto={len(auto_drop)}"
    )
    return metrics


def write_latex_table(results: Dict[str, Dict], out_tex: Path):
    # Build table rows
    def fmt_row(name: str, r: Dict) -> str:
        base = r["base"]; pl = r["platt"]; iso = r["isotonic"]
        # Choose calibrated values to display both Platt & Iso
        return (
            f"{name:9s} & "
            f"{base['auroc']:.4f} & "
            f"{pl['auroc']:.4f} / {iso['auroc']:.4f} & "
            f"{base['ece']:.4f} & {pl['ece']:.4f} / {iso['ece']:.4f} & "
            f"{base['brier']:.4f} & {pl['brier']:.4f} / {iso['brier']:.4f} \\\\"
        )

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Test performance across splits. Calibration fitted on validation predictions.}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "& \\multicolumn{2}{c}{AUROC} & \\multicolumn{2}{c}{ECE$\\downarrow$} & \\multicolumn{2}{c}{Brier$\\downarrow$} \\\\",
        "Split & Base & Platt / Iso & Base & Platt / Iso & Base & Platt / Iso \\\\",
        "\\midrule",
    ]
    for k in ("random", "temporal", "hospital"):
        if k in results:
            lines.append(fmt_row(k.capitalize(), results[k]))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LaTeX table -> {out_tex}")


def read_blocklist(path: Optional[Path]) -> set:
    if not path:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Blocklist file not found: {path}")
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return set(lines)


def main():
    ap = argparse.ArgumentParser(description="Thesis baseline: LGBM + calibration on custom cohorts")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Data root with YAIB-style Parquet")
    ap.add_argument("--splits", type=str, default=",".join(DEFAULT_SPLITS), help="Comma-separated splits to run")
    ap.add_argument("--label-candidates", type=str, default=",".join(DEFAULT_LABEL_CANDIDATES))
    ap.add_argument("--out", type=Path, default=OUT_ROOT, help="Output root for logs/plots/metrics")
    ap.add_argument("--ece-bins", type=int, default=DEFAULT_ECE_BINS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blocklist", type=Path, default=None, help="Optional text file with columns to drop (one per line)")
    args = ap.parse_args()

    root: Path = args.root
    out_root: Path = args.out
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    label_candidates = tuple(s.strip() for s in args.label_candidates.split(",") if s.strip())
    static_block = set(DEFAULT_STATIC_BLOCK) | read_blocklist(args.blocklist)

    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {}
    for split in splits:
        try:
            m = run_one_split(root, split, out_root, label_candidates, static_block, args.ece_bins, args.seed)
            results[split] = m
        except Exception as e:
            print(f"[ERROR] {split}: {e}")

    # Write a one-file CSV summary and LaTeX table
    rows = []
    for k, r in results.items():
        b, pl, iso = r["base"], r["platt"], r["isotonic"]
        rows.append(
            dict(
                split=k,
                features_n=r["features_n"],
                best_n_estimators=r["best_n_estimators"],
                base_auroc=b["auroc"],
                base_ece=b["ece"],
                base_brier=b["brier"],
                platt_auroc=pl["auroc"],
                platt_ece=pl["ece"],
                platt_brier=pl["brier"],
                isotonic_auroc=iso["auroc"],
                isotonic_ece=iso["ece"],
                isotonic_brier=iso["brier"],
            )
        )
    if rows:
        df = pd.DataFrame(rows)
        df.sort_values("split", inplace=True)
        (out_root / "summary.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        write_latex_table(results, out_root / "summary_table.tex")


if __name__ == "__main__":
    main()
