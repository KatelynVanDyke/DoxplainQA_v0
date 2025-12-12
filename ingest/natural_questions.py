from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path

OUTPUT = Path("../artifacts/natural_questions.parquet")
FORCE_REBUILD = False


def build():
    ds = load_dataset("natural_questions")
    rows = []

    for split in ds:

        if OUTPUT.exists() and not FORCE_REBUILD:
            print(f"[SKIP] {OUTPUT} already exists")
            return

        for ex in tqdm(ds[split], desc=f"natural_questions/{split}"):

            anns = ex.get("annotations")
            if not anns:
                continue

            short_answers = anns.get("short_answers", [])
            if not short_answers:
                continue

            sa = short_answers[0]
            starts = sa.get("start_token", [])
            ends = sa.get("end_token", [])

            if not starts or not ends:
                continue

            start, end = starts[0], ends[0]

            tokens = ex["document"]["tokens"]
            answer = " ".join(
                tokens[i]["token"]
                for i in range(start, end)
                if i in tokens
            ).strip()

            if not answer:
                continue

            context = ex["document"].get("text")
            if not context:
                continue

            rows.append({
                "dataset": "natural_questions",
                "id": ex["id"],
                "question": ex["question"]["text"].strip(),
                "answer": answer,
                "context": context.strip(),
            })

    pd.DataFrame(rows).to_parquet(OUTPUT, index=False)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    build()
