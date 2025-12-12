from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path

OUTPUT = Path("../artifacts/hotpotqa.parquet")
FORCE_REBUILD = False


def build():
    ds = load_dataset("hotpot_qa", "distractor")
    rows = []

    for split in ds:

        if OUTPUT.exists() and not FORCE_REBUILD:
            print(f"[SKIP] {OUTPUT} already exists")
            return

        for ex in tqdm(ds[split], desc=f"hotpotqa/{split}"):
            sf = ex["supporting_facts"]
            support_pairs = set()

            if isinstance(sf, dict):
                support_pairs = set(zip(sf["title"], sf["sent_id"]))
            else:
                for item in sf:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        support_pairs.add((item[0], item[1]))
                    elif isinstance(item, dict):
                        t = item.get("title")
                        i = item.get("sent_id")
                        if t is not None and i is not None:
                            support_pairs.add((t, i))

            if not support_pairs:
                continue

            context_parts = []
            context = ex["context"]

            for title, sents in zip(context["title"], context["sentences"]):
                for idx, sent in enumerate(sents):
                    if (title, idx) in support_pairs:
                        context_parts.append(sent)

            if not context_parts:
                continue

            rows.append({
                "dataset": "hotpotqa",
                "id": ex["id"],
                "question": ex["question"].strip(),
                "answer": ex["answer"].strip(),
                "context": " ".join(context_parts).strip(),
            })

    pd.DataFrame(rows).to_parquet(OUTPUT, index=False)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    build()
