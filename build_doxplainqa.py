import pandas as pd
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
OUTPUT = Path("doxplainqa.parquet")


def build():
    dfs = []

    for path in sorted(ARTIFACTS_DIR.glob("*.parquet")):
        df = pd.read_parquet(path)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    merged.to_parquet(OUTPUT, index=False)
    print(f"Saved → {OUTPUT}")
    print(f"Rows: {len(merged):,}")
    print(f"Datasets: {sorted(merged['dataset'].unique().tolist())}")


if __name__ == "__main__":
    build()
