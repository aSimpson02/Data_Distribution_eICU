# scripts/inspect_dataset.py
import pandas as pd
from pathlib import Path

IN = Path("data/all.parquet")

def main():
    df = pd.read_parquet(IN)
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nMissing % per column (top 20):")
    miss = df.isna().mean().sort_values(ascending=False) * 100
    print(miss.head(20).round(2))

    if "hospital_mortality" in df.columns:
        rate = df["hospital_mortality"].mean()
        print(f"\nMortality rate: {rate:.3f}")

    if "hospitalid" in df.columns:
        print("\nTop hospitals by count:")
        print(df["hospitalid"].value_counts().head(10))

    if "hospitaldischargeyear" in df.columns:
        print("\nCounts by discharge year:")
        print(df["hospitaldischargeyear"].value_counts().sort_index())

    print("\nNumeric summaries (selected):")
    num_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
    print(df[num_cols].describe(percentiles=[0.05,0.5,0.95]).T)

if __name__ == "__main__":
    main()
