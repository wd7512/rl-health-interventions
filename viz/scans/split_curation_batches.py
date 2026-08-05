"""Split refs into batches for parallel curation evaluation."""

import json
from pathlib import Path

CITATIONS_PATH = Path(__file__).parent.parent / "citations.json"
BATCHES_DIR = Path(__file__).parent / "curation_batches"


def split_into_batches():
    with open(CITATIONS_PATH) as f:
        data = json.load(f)

    refs = data.get("refs", {})
    ref_list = [{"id": k, **v} for k, v in refs.items()]

    BATCHES_DIR.mkdir(exist_ok=True)

    batch_size = 52
    batches = []
    for i in range(0, len(ref_list), batch_size):
        batch = ref_list[i : i + batch_size]
        batch_num = len(batches) + 1
        batch_file = BATCHES_DIR / f"batch_{batch_num:02d}.json"
        with open(batch_file, "w") as f:
            json.dump(batch, f, indent=2)
        batches.append(batch_file)
        print(f"Batch {batch_num}: {len(batch)} refs → {batch_file.name}")

    print(f"\nTotal: {len(ref_list)} refs split into {len(batches)} batches")
    return batches


if __name__ == "__main__":
    split_into_batches()
