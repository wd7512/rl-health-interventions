"""Merge scan results into citations.json."""

import json
from pathlib import Path

SCANS_DIR = Path(__file__).parent
CITATIONS_PATH = SCANS_DIR.parent / "citations.json"


def map_type(scan_type: str) -> str:
    """Map scan paper type to citations.json type field."""
    mapping = {
        "Trial": "curated",
        "Dataset": "curated",
        "Framework": "curated",
        "Simulator": "curated",
        "Benchmark": "curated",
        "Review": "ref",
    }
    return mapping.get(scan_type, "ref")


def merge_papers():
    with open(CITATIONS_PATH) as f:
        citations = json.load(f)

    with open(SCANS_DIR / "aggregated.json") as f:
        aggregated = json.load(f)

    existing_ids = set(citations["papers"].keys()) | set(citations["refs"].keys())
    scan_papers = aggregated.get("papers", [])
    added = 0
    skipped = 0

    for i, paper in enumerate(scan_papers):
        pid = paper.get("id", "")
        if not pid:
            skipped += 1
            continue

        if pid in existing_ids:
            skipped += 1
            continue

        ref_id = f"ref:scan:{added + 1:03d}"
        scan_type = paper.get("type", "Review")
        result = paper.get("result", "null")

        headline = paper.get("headline", "")
        if result == "negative" and headline and not headline.lower().startswith(("no ", "null", "fail", "negative")):
            headline = f"[Negative] {headline}"
        elif result == "mixed" and headline and not headline.lower().startswith(("mixed", "partial")):
            headline = f"[Mixed] {headline}"

        detail = paper.get("detail", "")
        notes = paper.get("notes", "")
        if notes:
            detail = f"{detail} {notes}".strip() if detail else notes

        missions = paper.get("missions", [])
        if missions:
            detail = f"{detail} (Found in missions: {', '.join(missions)})".strip() if detail else f"Found in missions: {', '.join(missions)}"

        ref_entry = {
            "label": paper.get("label", "")[:80],
            "authors": paper.get("authors", ""),
            "year": paper.get("year"),
            "venue": paper.get("venue", ""),
            "type": map_type(scan_type),
            "headline": headline[:200] if headline else "",
            "detail": detail[:500] if detail else "",
            "curated": scan_type in ["Trial", "Dataset", "Framework", "Simulator", "Benchmark"],
            "cites": [],
            "frontier": False,
            "evidence_score": 0.0,
            "def_rel": 0.0,
            "frontier_score": 0.0,
            "source_id": pid,
            "source_url": paper.get("source_url", ""),
            "scan_result": result,
            "scan_relevance": paper.get("relevance", "medium"),
        }

        citations["refs"][ref_id] = ref_entry
        existing_ids.add(pid)
        added += 1

    with open(CITATIONS_PATH, "w") as f:
        json.dump(citations, f, indent=2)

    print(f"Added {added} papers to citations.json")
    print(f"Skipped {skipped} papers (no ID or duplicate)")
    print(f"Total refs now: {len(citations['refs'])}")


if __name__ == "__main__":
    merge_papers()
