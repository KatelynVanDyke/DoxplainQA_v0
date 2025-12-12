from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path

OUTPUT = Path("../artifacts/boolq.parquet")
FORCE_REBUILD = False


def build():
    ds = load_dataset("boolq")
    rows = []

    for split in ds:

        if OUTPUT.exists() and not FORCE_REBUILD:
            print(f"[SKIP] {OUTPUT} already exists")
            return

        for ex in tqdm(ds[split], desc=f"boolq/{split}"):

            rows.append({
                "dataset": "boolq",
                "id": None,
                "question": ex["question"].strip(),
                "answer": str(ex["answer"]),
                "context": ex["passage"].strip(),
            })

    pd.DataFrame(rows).to_parquet(OUTPUT, index=False)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    build()
