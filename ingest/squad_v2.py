from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path

OUTPUT = Path("../artifacts/squad_v2.parquet")
FORCE_REBUILD = True


def build():
    ds = load_dataset("squad_v2")
    rows = []

    for split in ds:

        if OUTPUT.exists() and not FORCE_REBUILD:
            print(f"[SKIP] {OUTPUT} already exists")
            return

        for ex in tqdm(ds[split], desc=f"squad_v2/{split}"):
            answers = ex["answers"]["text"]
            if not answers:
                continue

            rows.append({
                "dataset": "squad_v2",
                "id": ex["id"],
                "question": ex["question"].strip(),
                "answer": answers[0].strip(),
                "context": ex["context"].strip(),
            })

    pd.DataFrame(rows).to_parquet(OUTPUT, index=False)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    build()
