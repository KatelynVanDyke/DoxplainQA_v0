from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path

OUTPUT = Path("../artifacts/narrativeqa.parquet")
FORCE_REBUILD = False


def build():
    ds = load_dataset("narrativeqa", "default")
    rows = []

    for split in ds:

        if OUTPUT.exists() and not FORCE_REBUILD:
            print(f"[SKIP] {OUTPUT} already exists")
            return

        for ex in tqdm(ds[split], desc=f"narrativeqa/{split}"):

            answers = ex.get("answers", [])
            if not answers:
                continue

            answer = answers[0].get("text")
            if not answer:
                continue

            context = ex.get("summary")
            if not context:
                continue

            rows.append({
                "dataset": "narrativeqa",
                "id": ex["id"],
                "question": ex["question"].strip(),
                "answer": answer.strip(),
                "context": context.strip(),
            })

    pd.DataFrame(rows).to_parquet(OUTPUT, index=False)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    build()
