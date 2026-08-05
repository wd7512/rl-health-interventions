"""Aggregate all mission JSON files into a single deduplicated file."""

import json
from pathlib import Path

SCANS_DIR = Path(__file__).parent


def main():
    all_papers = {}
    mission_summaries = {}

    for mf in sorted(SCANS_DIR.glob("mission_*.json")):
        with open(mf) as f:
            data = json.load(f)
        mission_id = data.get("mission_id", mf.stem)
        topic = data.get("topic", "")
        papers = data.get("papers", [])
        summary = data.get("summary", {})

        mission_summaries[mission_id] = {
            "topic": topic,
            "total": summary.get("total_papers", len(papers)),
            "positive": summary.get("positive_results", 0),
            "negative": summary.get("negative_results", 0),
            "mixed": summary.get("mixed_results", 0),
            "key_findings": summary.get("key_findings", []),
        }

        for p in papers:
            pid = p.get("id", "")
            if not pid:
                continue
            if pid in all_papers:
                all_papers[pid]["missions"].append(mission_id)
            else:
                all_papers[pid] = {**p, "missions": [mission_id]}

    aggregated = {
        "scan_date": "2026-01-13",
        "total_unique_papers": len(all_papers),
        "missions": mission_summaries,
        "papers": list(all_papers.values()),
    }

    output_path = SCANS_DIR / "aggregated.json"
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Aggregated {len(all_papers)} unique papers from {len(mission_summaries)} missions")
    print(f"Written to {output_path}")

    by_mission_count = {}
    for p in all_papers.values():
        n = len(p["missions"])
        by_mission_count[n] = by_mission_count.get(n, 0) + 1
    print(f"Papers appearing in multiple missions: {sum(v for k, v in by_mission_count.items() if k > 1)}")


if __name__ == "__main__":
    main()
