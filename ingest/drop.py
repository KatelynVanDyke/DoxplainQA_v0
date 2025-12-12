from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path

OUTPUT = Path("../artifacts/drop.parquet")
FORCE_REBUILD = False


def build():
    ds = load_dataset("drop")
    rows = []

    for split in ds:

        if OUTPUT.exists() and not FORCE_REBUILD:
            print(f"[SKIP] {OUTPUT} already exists")
            return

        for ex in tqdm(ds[split], desc=f"drop/{split}"):

            answers = ex.get("answers", [])
            if not answers:
                continue

            answer = answers[0]
            if not answer:
                continue

            context = ex.get("passage")
            if not context:
                continue

            rows.append({
                "dataset": "drop",
                "id": ex["query_id"],
                "question": ex["question"].strip(),
                "answer": answer.strip(),
                "context": context.strip(),
            })

    pd.DataFrame(rows).to_parquet(OUTPUT, index=False)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    build()
