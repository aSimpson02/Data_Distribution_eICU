#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
import gzip, shutil

CSV_DIR_DEFAULT = Path("data/csv_splits")

# never use these as predictors (leakage / IDs)
LEAKY_COLS = {
    "patientunitstayid",
    "hospitalid",
    "hospitaldischargeyear",
    "apachescore",
    "predictedhospitalmortality",
    "admissionoffset",
}

def _maybe_decompress(src_gz: Path, dst_csv: Path) -> Path:
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src_gz, "rt") as fin, open(dst_csv, "w", newline="") as fout:
        shutil.copyfileobj(fin, fout)
    return dst_csv

def _candidate_paths(base: Path, stem: str) -> list[Path]:
    # stem examples: "random_train", "temporal_val", "hospital_test"
    split = stem.split("_")[0]
    part  = stem.split("_")[-1]
    return [
        base / f"{stem}.csv",
        base / f"{stem}.csv.gz",
        base / split / f"{part}.csv",
        base / split / f"{part}.csv.gz",
    ]

def _read_one(base: Path, stem: str) -> pd.DataFrame:
    cands = _candidate_paths(base, stem)
    for p in cands:
        if p.exists():
            if p.suffix == ".gz":
                # decompress to temp and read
                tmp = base / f"__tmp__{stem}.csv"
                out = _maybe_decompress(p, tmp)
                df = pd.read_csv(out)
                try: tmp.unlink(missing_ok=True)
                except Exception: pass
                break
            else:
                df = pd.read_csv(p)
                break
    else:
        raise FileNotFoundError(f"Missing split file for {stem} under {base}")

    # sanitize columns once (fix whitespace warning)
    df.columns = df.columns.str.replace(r"\s+", "_", regex=True)
    return df

def load_three(stem_base: str, splits_dir: str | Path = CSV_DIR_DEFAULT):
    d = Path(splits_dir)
    tr = _read_one(d, f"{stem_base}_train")
    va = _read_one(d, f"{stem_base}_val")
    te = _read_one(d, f"{stem_base}_test")
    return tr, va, te

def drop_leaky(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    keep = [c for c in df.columns if (c == label_col) or (c not in LEAKY_COLS)]
    return df[keep]
