# scripts/make_cohorts.py
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ID = "patientunitstayid"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def build_preproc(cat, num):
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    return ColumnTransformer([("num", num_pipe, num),
                              ("cat", cat_pipe, cat)], remainder="drop")

def prepare_features(df, target_col, meta_map, categorical, numeric):
    meta = pd.DataFrame({
        "hospital": df[meta_map["hospital"]],
        "sex": df[meta_map["sex"]],
        "age": df[meta_map["age"]],
        "time": df[meta_map["time"]],
    })
    y = df[target_col].astype(int).to_numpy()
    used = [c for c in (numeric + categorical) if c in df.columns and c != target_col]
    return df[used], y, meta, used

def save_prefix(prefix, X, y, meta, out_dir: Path):
    ensure_dir(out_dir)
    pd.DataFrame(X).to_parquet(out_dir/f"{prefix}_X.parquet", index=False)
    pd.DataFrame({"y": y}).to_parquet(out_dir/f"{prefix}_y.parquet", index=False)
    meta.to_parquet(out_dir/f"{prefix}_meta.parquet", index=False)

def elig_hospitals(df, min_n=1000):
    cnt = df.groupby("hospitalid")[ID].nunique().rename("n").reset_index()
    return set(cnt.loc[cnt["n"]>=min_n, "hospitalid"].astype(int).tolist())

def pick_sources_target(eligible, target):
    eligible = list(sorted(eligible))
    if target not in eligible:
        raise ValueError(f"Target {target} not in eligible list")
    src = [h for h in eligible if h != target]
    if len(src) < 5:
        raise ValueError("Need at least 6 eligible hospitals")
    return src[:5], target

def temporal_mask(meta, start, end, time_col="time"):
    t = pd.to_datetime(meta[time_col], errors="coerce")
    # admissionoffset is minutes → convert if numeric
    if pd.api.types.is_numeric_dtype(meta[time_col]):
        base = pd.Timestamp("2014-01-01")
        t = base + pd.to_timedelta(meta[time_col].astype(float), unit="m")
    return (t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", default="data/all.parquet")
    ap.add_argument("--out_dir", default="data/processed/cohorts")
    ap.add_argument("--min_n", type=int, default=1000)
    ap.add_argument("--target_col", default="hospital_mortality")
    ap.add_argument("--meta_hospital", default="hospitalid")
    ap.add_argument("--meta_sex", default="gender")
    ap.add_argument("--meta_age", default="age")
    ap.add_argument("--meta_time", default="admissionoffset")
    ap.add_argument("--categorical", nargs="*", default=["gender","diagnosisstring"])
    ap.add_argument("--numeric", nargs="*", default=[
        "age","apachescore","heartrate","systemicsystolic","systemicdiastolic","spo2",
        "albumin","bun","glucose","lactate","bilirubin"])
    ap.add_argument("--inter", default="75,420,58,12,33->188")  # src5->tgt
    ap.add_argument("--temporal", default="2014-01-01,2014-12-31->2015-01-01,2015-12-31")
    args = ap.parse_args()

    df = pd.read_parquet(args.all)
    df = df.rename(columns={args.meta_hospital:"hospitalid",
                            args.meta_sex:"gender",
                            args.meta_age:"age",
                            args.meta_time:"admissionoffset"})
    eligible = elig_hospitals(df, args.min_n)
    print(f"Eligible hospitals (>= {args.min_n} stays): {len(eligible)}")

    # feature prep (fit on train ONLY later)
    meta_map = {"hospital":"hospitalid","sex":"gender","age":"age","time":"admissionoffset"}
    Xraw, y_all, meta_all, used = prepare_features(
        df, args.target_col, meta_map, args.categorical, args.numeric
    )

    # INTER-HOSPITAL
    src5_str, tgt_str = args.inter.split("->")
    src5 = [int(x) for x in src5_str.split(",")]
    tgt = int(tgt_str)
    for h in src5+[tgt]:
        if h not in eligible:
            raise ValueError(f"Hospital {h} not eligible or missing")

    h = df["hospitalid"].astype(int).to_numpy()
    tr_mask = np.isin(h, np.array(src5))
    te_mask = (h == tgt)

    # Train/val split within sources (stratified)
    X_src, y_src = Xraw.loc[tr_mask].reset_index(drop=True), y_all[tr_mask]
    meta_src = meta_all.loc[tr_mask].reset_index(drop=True)
    X_tr, X_val, y_tr, y_val, m_tr, m_val = train_test_split(
        X_src, y_src, meta_src, test_size=0.2, stratify=y_src, random_state=42
    )
    X_te, y_te, m_te = Xraw.loc[te_mask].reset_index(drop=True), y_all[te_mask], meta_all.loc[te_mask].reset_index(drop=True)

    # Fit preprocessor on TRAIN ONLY, transform others
    ct = build_preproc(args.categorical, args.numeric)
    Xtr = ct.fit_transform(X_tr)
    Xval = ct.transform(X_val)
    Xte  = ct.transform(X_te)

    out = Path(args.out_dir); ensure_dir(out)
    save_prefix("ih_train", Xtr, y_tr, m_tr, out)
    save_prefix("ih_val",   Xval, y_val, m_val, out)
    save_prefix("ih_test",  Xte,  y_te,  m_te,  out)
    print("Saved inter-hospital cohorts →", out)

    # TEMPORAL
    (src_s, src_e), (tgt_s, tgt_e) = [tuple(s.split(",")) for s in args.temporal.split("->")]
    src_mask = temporal_mask(meta_all, src_s, src_e)
    tgt_mask = temporal_mask(meta_all, tgt_s, tgt_e)

    X_src, y_src = Xraw.loc[src_mask].reset_index(drop=True), y_all[src_mask]
    meta_src = meta_all.loc[src_mask].reset_index(drop=True)
    X_tr, X_val, y_tr, y_val, m_tr, m_val = train_test_split(
        X_src, y_src, meta_src, test_size=0.2, stratify=y_src, random_state=42
    )
    X_te, y_te, m_te = Xraw.loc[tgt_mask].reset_index(drop=True), y_all[tgt_mask], meta_all.loc[tgt_mask].reset_index(drop=True)

    ct2 = build_preproc(args.categorical, args.numeric)
    Xtr = ct2.fit_transform(X_tr)
    Xval = ct2.transform(X_val)
    Xte  = ct2.transform(X_te)

    save_prefix("temp_train", Xtr, y_tr, m_tr, out)
    save_prefix("temp_val",   Xval, y_val, m_val, out)
    save_prefix("temp_test",  Xte,  y_te,  m_te,  out)
    print("Saved temporal cohorts →", out)

if __name__ == "__main__":
    main()
