#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, chi2_contingency, norm, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------- discovery ----------

def find_candidate_split_dirs(root: str):
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(f"--root not found: {root}")
    cands = []
    for d in rootp.glob("**/"):
        if not d.is_dir():
            continue
        if d == rootp:
            continue
        csvs = list(d.rglob("*.csv"))
        names = [p.name.lower() for p in csvs]
        if any(("train" in n or "trn" in n) for n in names) and any(("test" in n or "tst" in n) for n in names):
            cands.append(d)
    uniq, seen = [], set()
    for d in sorted(cands, key=lambda p: len(str(p))):
        if str(d) not in seen:
            uniq.append(d); seen.add(str(d))
    return uniq

def pick_split_files(split_dir: Path):
    csvs = [p for p in split_dir.rglob("*.csv")]
    def pick(tokens):
        cand = [p for p in csvs if any(t in p.name.lower() for t in tokens)]
        if not cand:
            return None
        cand.sort(key=lambda p: (p.name.lower() != f"{tokens[0]}.csv", len(str(p))))
        return str(cand[0])
    train = pick(["train","trn"]); test = pick(["test","tst"]); val = pick(["val","valid","validation"])
    if train is None or test is None:
        raise FileNotFoundError(f"No train/test CSV under {split_dir}")
    return {"train": train, "val": val, "test": test}

# ---------- stats ----------

def _safe_hist(a, b, bins=30, eps=1e-12):
    p, edges = np.histogram(a, bins=bins, density=True)
    q, _     = np.histogram(b, bins=edges, density=True)
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    return p, q

def js_divergence_hist(a, b, bins=30):
    """
    Continuous JS divergence via histogram then SciPy Jensen–Shannon distance (base 2).
    JS divergence = JS distance^2, bounded [0,1].
    """
    p, q = _safe_hist(a, b, bins=bins, eps=1e-12)
    jsd = jensenshannon(p, q, base=2)
    return float(jsd**2)

def js_divergence_probs(p, q):
    """JS divergence on discrete probabilities p,q (already normalised)."""
    jsd = jensenshannon(p, q, base=2)
    return float(jsd**2)

def entropy(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def entropy_shift(p_src, p_tgt):
    return float(np.mean(entropy(p_tgt)) - np.mean(entropy(p_src)))

def two_prop_z_test(n1, k1, n2, k2):
    p1, p2 = k1/n1, k2/n2
    p_pool = (k1+k2)/(n1+n2)
    se = np.sqrt(p_pool*(1-pool)*(1/n1 + 1/n2)) if (pool:=p_pool) or True else None  # placeholder to avoid lints
    se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    if se == 0: return (p1,p2,p2-p1), 1.0
    z = (p1 - p2) / se
    pval = 2*(1 - norm.cdf(abs(z)))
    return (p1,p2,p2-p1), pval

def bh_fdr(pvals, q=0.05):
    p = np.asarray(pvals, dtype=float)
    n = np.sum(~np.isnan(p))
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(p)+1)
    thresh = (ranks / max(n,1)) * q
    disc = p <= thresh
    if np.any(disc):
        kmax = np.max(np.where(disc)[0])
        disc = np.zeros_like(p, dtype=bool)
        disc[order[:kmax+1]] = True
    else:
        disc = np.zeros_like(p, dtype=bool)
    return disc

# ---------- computations ----------

POSSIBLE_LABELS = [
    "label","y","y_true","target","mortality","hospital_mortality",
    "in_hospital_mortality","hospital_expire_flag","death","outcome"
]

def get_label_column(df: pd.DataFrame, preferred: str):
    cols = [c.lower() for c in df.columns]
    if preferred in df.columns:
        return preferred
    if preferred.lower() in cols:
        return df.columns[cols.index(preferred.lower())]
    for name in POSSIBLE_LABELS:
        if name in cols:
            return df.columns[cols.index(name)]
    for c in df.columns:
        if any(k in c.lower() for k in ("mort","expire","death","outcome")):
            nunq = df[c].dropna().nunique()
            if nunq <= 3:
                return c
    raise KeyError(f"Could not find label column. Tried: { [preferred] + POSSIBLE_LABELS } ; got: {list(df.columns)}")

def infer_feature_types(df, label_col, ignore_cols):
    types = {}
    for c in df.columns:
        if c == label_col or c in ignore_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            nunq = df[c].nunique(dropna=True)
            if nunq <= 10 and set(df[c].dropna().unique()).issubset(set(range(11))):
                types[c] = "cat"
            else:
                types[c] = "cont"
        else:
            types[c] = "cat"
    return types

def feature_shift_table(train_df, test_df, feature_types, bins=30):
    rows = []
    for col, ftype in feature_types.items():
        tr = train_df[col].dropna().values
        te = test_df[col].dropna().values
        if len(tr)==0 or len(te)==0:
            continue
        if ftype == "cont":
            ks_stat, ks_p = ks_2samp(tr, te)
            js = js_divergence_hist(tr, te, bins=bins)
            w  = wasserstein_distance(tr, te)
            rows.append({"feature": col, "type":"cont", "KS": ks_stat, "KS_p": ks_p, "JS": js, "Wasserstein": w})
        else:
            tr_ct = pd.Series(tr).value_counts()
            te_ct = pd.Series(te).value_counts()
            cats = sorted(set(tr_ct.index).union(te_ct.index))
            obs = np.vstack([tr_ct.reindex(cats, fill_value=0).values,
                             te_ct.reindex(cats, fill_value=0).values])
            try:
                chi2, chi_p, *_ = chi2_contingency(obs)
            except ValueError:
                chi2, chi_p = np.nan, 1.0
            p = obs[0] / max(obs[0].sum(), 1)
            q = obs[1] / max(obs[1].sum(), 1)
            js = js_divergence_probs(p, q)
            rows.append({"feature": col, "type":"cat", "chi2": chi2, "chi2_p": chi_p, "JS": js})
    df = pd.DataFrame(rows)
    if "KS_p" in df:
        mask = df["type"].eq("cont")
        if mask.any():
            sig = np.full(df.shape[0], False); idx = df.index[mask]
            sig[idx] = bh_fdr(df.loc[mask,"KS_p"].values, q=0.05)
            df["KS_sig_FDR"] = sig
    if "chi2_p" in df:
        mask = df["type"].eq("cat")
        if mask.any():
            sig = np.full(df.shape[0], False); idx = df.index[mask]
            sig[idx] = bh_fdr(df.loc[mask,"chi2_p"].values, q=0.05)
            df["CHI2_sig_FDR"] = sig
    return df

def missingness_drift(train_df, test_df, label_col):
    rows=[]
    for col in train_df.columns:
        if col == label_col:
            continue
        tm = train_df[col].isna().mean(); sm = test_df[col].isna().mean()
        rows.append({"feature":col,"train_missing":tm,"test_missing":sm,"delta_missing":sm-tm})
    return pd.DataFrame(rows).sort_values("delta_missing", ascending=False)

def shift_signature(ftab):
    mean_js = ftab["JS"].mean()
    mean_ks = ftab.loc[ftab["type"]=="cont","KS"].mean() if "KS" in ftab else np.nan
    mean_w  = ftab["Wasserstein"].mean() if "Wasserstein" in ftab else np.nan
    return dict(mean_JS=float(mean_js),
                mean_KS_cont=(float(mean_ks) if mean_ks==mean_ks else None),
                mean_Wasserstein=(float(mean_w) if mean_w==mean_w else None))

def pca_plot(train_X, test_X, outpath, title):
    X = np.vstack([train_X, test_X])
    X = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2, random_state=0).fit_transform(X)
    n = train_X.shape[0]
    plt.figure(figsize=(4.2,3.6))
    plt.scatter(Z[:n,0], Z[:n,1], s=6, alpha=0.55, label="source")
    plt.scatter(Z[n:,0], Z[n:,1], s=6, alpha=0.55, label="target")
    plt.legend(); plt.title(title); plt.tight_layout()
    plt.savefig(outpath, dpi=220); plt.close()

# ---------- one split ----------

def run_one_split(split_name: str, split_dir: Path, preds_csv: str, outdir: Path, label_col="label", ignore_cols=("id",)):
    outdir.mkdir(parents=True, exist_ok=True)
    files = pick_split_files(split_dir)
    tr = pd.read_csv(files["train"])
    te = pd.read_csv(files["test"])

    # label detection (or confirm)
    try:
        lbl = get_label_column(tr, label_col)
    except KeyError as e:
        raise KeyError(f"[{split_name}] {e}")
    if lbl != label_col:
        print(f"[INFO] {split_name}: using detected label column '{lbl}'")
    label_col = lbl

    ftypes = infer_feature_types(tr, label_col=label_col, ignore_cols=ignore_cols)
    ftab = feature_shift_table(tr, te, ftypes, bins=30)
    ftab.to_csv(outdir / f"{split_name}_feature_shift.csv", index=False)

    miss = missingness_drift(tr, te, label_col=label_col)
    miss.to_csv(outdir / f"{split_name}_missingness_drift.csv", index=False)

    sig = shift_signature(ftab)
    n1, k1 = len(tr), int(tr[label_col].sum())
    n2, k2 = len(te), int(te[label_col].sum())
    (p_src, p_tgt, dlt), pval = two_prop_z_test(n1,k1,n2,k2)
    sig.update({"label_p_src":float(p_src),"label_p_tgt":float(p_tgt),"label_delta":float(dlt),"label_pval":float(pval)})

    # entropy shift per model (optional)
    if preds_csv and os.path.exists(preds_csv):
        preds = pd.read_csv(preds_csv)
        preds.columns = [c.strip().lower() for c in preds.columns]
        need = {"split","set","model","y_pred_proba"}
        ent_rows=[]
        if need.issubset(set(preds.columns)):
            valp = preds[(preds["split"].str.lower()==split_name) & (preds["set"].str.lower()=="val")]
            tstp = preds[(preds["split"].str.lower()==split_name) & (preds["set"].str.lower()=="test")]
            for m in sorted(tstp["model"].unique()):
                pv = valp[valp["model"]==m]["y_pred_proba"].values
                pt = tstp[tstp["model"]==m]["y_pred_proba"].values
                if len(pv)>0 and len(pt)>0:
                    ent_rows.append({"scenario":split_name,"model":m,"entropy_shift":entropy_shift(pv,pt)})
        if ent_rows:
            pd.DataFrame(ent_rows).to_csv(outdir / f"{split_name}_entropy_shift_by_model.csv", index=False)

    pd.DataFrame([{"scenario":split_name, **sig}]).to_csv(outdir / f"{split_name}_shift_signature.csv", index=False)
    with open(outdir / f"{split_name}_shift_signature.json","w") as f:
        json.dump({"scenario":split_name, **sig}, f, indent=2)

    top = ftab.sort_values("JS", ascending=False).head(20).reset_index(drop=True)
    top.to_csv(outdir / f"{split_name}_topdrifting.csv", index=False)

    cont_cols = [c for c,t in ftypes.items() if t=="cont"]
    if len(cont_cols)>=2:
        pca_plot(
            tr[cont_cols].fillna(tr[cont_cols].median()).values,
            te[cont_cols].fillna(tr[cont_cols].median()).values,
            outdir / f"{split_name}_pca.png",
            f"PCA: {split_name} (source vs target)"
        )

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--preds", default="models/preds/preds_all.csv")
    ap.add_argument("--out", default="results/shift")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--ignore_cols", nargs="*", default=["id"])
    ap.add_argument("--splits", nargs="*", default=None,
                    help="Optional split name filters (case-insensitive substring match)")
    args = ap.parse_args()

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    cand_dirs = find_candidate_split_dirs(args.root)
    if args.splits:
        wanted = [s.lower() for s in args.splits]
        cand_dirs = [d for d in cand_dirs if any(w in d.name.lower() or w in str(d).lower() for w in wanted)]

    if not cand_dirs:
        raise FileNotFoundError(f"No split directories with train/test CSVs found under {args.root}")

    print("[INFO] Found split dirs:")
    for d in cand_dirs: print("  -", d)

    for d in cand_dirs:
        split_name = d.name.lower()
        run_one_split(
            split_name=split_name,
            split_dir=d,
            preds_csv=args.preds if os.path.exists(args.preds) else None,
            outdir=out_root / split_name,
            label_col=args.label_col,
            ignore_cols=tuple(args.ignore_cols)
        )

if __name__ == "__main__":
    main()
