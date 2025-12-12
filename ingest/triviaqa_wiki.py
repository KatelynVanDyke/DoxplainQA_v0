from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path

OUTPUT = Path("../artifacts/triviaqa.parquet")
FORCE_REBUILD = False


def build():
    ds = load_dataset("trivia_qa", "rc.wikipedia")
    rows = []

    for split in ds:

        if OUTPUT.exists() and not FORCE_REBUILD:
            print(f"[SKIP] {OUTPUT} already exists")
            return

        for ex in tqdm(ds[split], desc=f"triviaqa/{split}"):

            answer = ex.get("answer", {}).get("value")
            if not answer:
                continue

            contexts = ex.get("entity_pages", {}).get("wiki_context", [])
            if not contexts:
                continue

            rows.append({
                "dataset": "triviaqa",
                "id": ex.get("question_id"),
                "question": ex["question"].strip(),
                "answer": answer.strip(),
                "context": " ".join(contexts).strip(),
            })

    pd.DataFrame(rows).to_parquet(OUTPUT, index=False)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    build()
