# scripts/list_headers.py
import pandas as pd
from pathlib import Path
import gzip

RAW = Path("eicu-collaborative-research-database-2.0")
OUT = Path("data"); OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "file_headers.txt"

def main():
    with REPORT.open("w") as f:
        for file in RAW.glob("*.csv.gz"):
            try:
                # get header directly from gzip (first line)
                with gzip.open(file, "rt") as gz:
                    header_line = gz.readline().strip()

                # also preview with pandas
                df = pd.read_csv(file, nrows=5)  # small sample
                cols = df.columns.tolist()

                # fast row count (approx)
                nrows = sum(1 for _ in gzip.open(file, "rt")) - 1

                f.write(f"{file.name}:\n")
                f.write(f"  Rows: ~{nrows:,}\n")
                f.write(f"  Header line: {header_line}\n")
                f.write(f"  Columns ({len(cols)}): {cols}\n\n")

                print(f"[OK] {file.name} ({len(cols)} cols, ~{nrows:,} rows)")
            except Exception as e:
                f.write(f"{file.name}: ERROR {e}\n")
                print(f"[ERROR] {file.name}: {e}")

    print(f"\nReport saved → {REPORT}")

if __name__ == "__main__":
    main()
