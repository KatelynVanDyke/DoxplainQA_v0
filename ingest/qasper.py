from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path

OUTPUT = Path("../artifacts/qasper.parquet")
FORCE_REBUILD = False


def build():
    ds = load_dataset("allenai/qasper")
    rows = []

    for split in ds:

        if OUTPUT.exists() and not FORCE_REBUILD:
            print(f"[SKIP] {OUTPUT} already exists")
            return

        for ex in tqdm(ds[split], desc=f"qasper/{split}"):

            answers = ex.get("answers", [])
            if not answers:
                continue

            ans = answers[0]
            answer_text = ans.get("answer")
            evidence = ans.get("evidence", [])

            if not answer_text or not evidence:
                continue

            context_parts = []
            for ev in evidence:
                text = ev.get("text")
                if text:
                    context_parts.append(text)

            if not context_parts:
                continue

            rows.append({
                "dataset": "qasper",
                "id": ex["id"],
                "question": ex["question"].strip(),
                "answer": answer_text.strip(),
                "context": " ".join(context_parts).strip(),
            })

    pd.DataFrame(rows).to_parquet(OUTPUT, index=False)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    build()
