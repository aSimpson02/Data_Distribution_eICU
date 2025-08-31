# scripts/build_all_from_eicu.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

RAW = Path("eicu-collaborative-research-database-2.0")
OUT = Path("data"); OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "missing_columns.log"

ID = "patientunitstayid"

# Labs wanted from lab.csv (long format)
LAB_KEEP = {
    "albumin", "bun", "glucose", "lactate", "bilirubin", "creatinine"
}

# Vital columns wanted - check if present!!
VITAL_CANDS = [
    "heartrate", "systemicsystolic", "systemicdiastolic", "systemicmean",
    "sao2", "respiration", "temperature"
]


def _log_missing(file, missing):
    if not missing:
        return
    with LOG.open("a") as f:
        f.write(f"{file}: missing {sorted(list(missing))}\n")
    print(f"[WARN] {file} missing {sorted(list(missing))}")


def agg_chunked_numeric_means(path: Path, key: str, cols: list[str], chunksize=500_000):
    """Chunked read of wide numeric table → per-patient mean for provided cols."""
    found = pd.read_csv(path, nrows=0).columns
    keep = [c for c in cols if c in found]
    if key not in keep:
        keep = [key] + keep
    missing = set(cols) - set(keep)
    _log_missing(path.name, missing)

    parts = []
    for ch in pd.read_csv(path, usecols=keep, chunksize=chunksize, low_memory=False):
        grp = ch.groupby(key, dropna=False).mean(numeric_only=True)
        parts.append(grp)
    if not parts:
        return pd.DataFrame({key: []})
    df = pd.concat(parts).groupby(level=0).mean(numeric_only=True).reset_index()
    return df


def labs_long_to_wide(path: Path, key: str, chunksize=1_000_000):
    """Read lab.csv(long): filter LAB_KEEP, pivot to wide columns (mean per patient)."""
    found = pd.read_csv(path, nrows=0).columns
    need = [key, "labname", "labresult"]
    keep = [c for c in need if c in found]
    missing = set(need) - set(keep)
    _log_missing(path.name, missing)
    if not all(c in keep for c in [key, "labname", "labresult"]):
        # Return empty if critical columns = missing
        return pd.DataFrame({key: []})

    parts = []
    for ch in pd.read_csv(path, usecols=keep, chunksize=chunksize, low_memory=False):
        # normaliing names
        ch["labname_norm"] = ch["labname"].astype(str).str.lower().str.strip()
        ch = ch[ch["labname_norm"].isin(LAB_KEEP)]
        if ch.empty:
            continue
        # average result per patient & lab
        grp = ch.groupby([key, "labname_norm"], dropna=False)["labresult"] \
                .mean().reset_index()
        parts.append(grp)

    if not parts:
        return pd.DataFrame({key: []})

    long_df = pd.concat(parts, axis=0)
    wide = long_df.pivot_table(index=key, columns="labname_norm",
                               values="labresult", aggfunc="mean").reset_index()
    # checking expected columns exist
    for col in LAB_KEEP:
        if col not in wide.columns:
            wide[col] = pd.NA
    return wide[[key] + sorted(list(LAB_KEEP))]


def main():
    if LOG.exists():
        LOG.unlink()

    # 1) PATIENT (core demographics, shift/meta, outcome)
    print("1) patient …")
    patient_cols = [
        ID, "hospitalid", "gender", "age",
        "unitadmissionoffset", "hospitaladmitoffset",
        "hospitaldischargestatus", "hospitaldischargeyear",
        "unittype"
    ]
    found = pd.read_csv(RAW/"patient.csv.gz", nrows=0).columns
    keep = [c for c in patient_cols if c in found]
    missing = set(patient_cols) - set(keep)
    _log_missing("patient.csv.gz", missing)

    patient = pd.read_csv(RAW/"patient.csv.gz", usecols=keep, low_memory=False)

    # label
    if "hospitaldischargestatus" in patient.columns:
        patient["hospital_mortality"] = (patient["hospitaldischargestatus"] == "Expired").astype(int)
    else:
        patient["hospital_mortality"] = pd.NA

    # time proxy: prefer unitadmissionoffset, else hospitaladmitoffset, else 0
    if "unitadmissionoffset" in patient.columns:
        patient = patient.rename(columns={"unitadmissionoffset": "admissionoffset"})
    elif "hospitaladmitoffset" in patient.columns:
        patient = patient.rename(columns={"hospitaladmitoffset": "admissionoffset"})
    else:
        patient["admissionoffset"] = 0

    # clean age 
    if "age" in patient.columns:
        patient["age"] = (
            patient["age"]
            .astype(str)
            .str.replace(r"[^0-9]", "", regex=True)
        )
        patient["age"] = pd.to_numeric(patient["age"], errors="coerce")

    # 2) APACHE (severity & predicted risk)
    print("2) apachePatientResult …")
    apache_wide = agg_chunked_numeric_means(
        RAW/"apachePatientResult.csv.gz",
        key=ID,
        cols=[ID, "apachescore", "predictedhospitalmortality"],
        chunksize=200_000
    )

    # 3) DIAGNOSIS 
    print("3) diagnosis …")
    diag_found = pd.read_csv(RAW/"diagnosis.csv.gz", nrows=0).columns
    diag_keep = [c for c in [ID, "diagnosisstring", "diagnosisoffset"] if c in diag_found]
    _log_missing("diagnosis.csv.gz", set([ID, "diagnosisstring", "diagnosisoffset"]) - set(diag_keep))

    # take the earliest diagnosis per stay if offset exists; else just first
    diag_iter = pd.read_csv(RAW/"diagnosis.csv.gz", usecols=diag_keep,
                            chunksize=1_000_000, low_memory=False)
    diag_parts = []
    for ch in diag_iter:
        if "diagnosisoffset" in ch.columns:
            ch = ch.sort_values(["patientunitstayid", "diagnosisoffset"]).groupby(ID, as_index=False).first()
        else:
            ch = ch.groupby(ID, as_index=False).first()
        diag_parts.append(ch[[ID, "diagnosisstring"]])
    diagnosis = pd.concat(diag_parts, axis=0).drop_duplicates(subset=[ID])

    # 4) VITALS 
    print("4) vitalPeriodic (mean) …")
    vitals_wide = agg_chunked_numeric_means(
        RAW/"vitalPeriodic.csv.gz",
        key=ID,
        cols=[ID] + VITAL_CANDS,
        chunksize=1_000_000
    )
    # rename sao2 
    if "sao2" in vitals_wide.columns and "spo2" not in vitals_wide.columns:
        vitals_wide = vitals_wide.rename(columns={"sao2": "spo2"})

    # 5) LABS 
    print("5) lab (long→wide means) …")
    labs_wide = labs_long_to_wide(RAW/"lab.csv.gz", key=ID, chunksize=1_000_000)

    # Merge everything!
    print("Merging …")
    merged = (patient
              .merge(apache_wide, on=ID, how="left")
              .merge(diagnosis, on=ID, how="left")
              .merge(vitals_wide, on=ID, how="left")
              .merge(labs_wide, on=ID, how="left"))


    final_cols = [c for c in [
        ID, "hospitalid", "unittype", "gender", "age",
        "admissionoffset", "hospitaldischargeyear", "hospital_mortality",
        # Apache
        "apachescore", "predictedhospitalmortality",
        # Diagnosis
        "diagnosisstring",
        # Vitals
        "heartrate", "systemicsystolic", "systemicdiastolic", "systemicmean",
        "respiration", "temperature", "spo2",
        # Labs (wide)
        "albumin", "bun", "glucose", "lactate", "bilirubin", "creatinine",
    ] if c in merged.columns]

    merged = merged[final_cols].copy()

    print("Final shape:", merged.shape)
    merged.to_parquet(OUT/"all.parquet", index=False)
    print("Saved → data/all.parquet")

    if LOG.exists():
        print(f"[LOG] Missing column info written to {LOG}")


if __name__ == "__main__":
    main()
