#!/usr/bin/env python3
"""
Aggregate curation batch evaluations and apply to citations.json
"""
import json
from pathlib import Path

def aggregate_evaluations():
    """Read all batch evaluation files and aggregate decisions"""
    batches_dir = Path("viz/scans/curation_batches")
    eval_files = sorted(batches_dir.glob("batch_*_evaluation.json"))

    all_promotions = []
    all_demotions = []
    summary = {
        "total_evaluated": 0,
        "promoted": 0,
        "demoted": 0,
        "kept_curated": 0,
        "kept_regular": 0
    }

    for eval_file in eval_files:
        with open(eval_file) as f:
            data = json.load(f)

        all_promotions.extend(data.get("promotions", []))
        all_demotions.extend(data.get("demotions", []))

        batch_summary = data.get("summary", {})
        summary["total_evaluated"] += batch_summary.get("total_evaluated", 0)
        summary["promoted"] += batch_summary.get("promoted", 0)
        summary["demoted"] += batch_summary.get("demoted", 0)
        summary["kept_curated"] += batch_summary.get("kept_curated", 0)
        summary["kept_regular"] += batch_summary.get("kept_regular", 0)

    return {
        "promotions": all_promotions,
        "demotions": all_demotions,
        "summary": summary
    }

def apply_curation_changes(aggregated):
    """Apply promotions and demotions to citations.json"""
    citations_path = Path("viz/citations.json")

    with open(citations_path) as f:
        citations = json.load(f)

    refs = citations.get("refs", {})
    changes = {"promoted": [], "demoted": [], "errors": []}

    # Apply promotions
    for promo in aggregated["promotions"]:
        ref_id = promo["id"]
        if ref_id in refs:
            if not refs[ref_id].get("curated", False):
                refs[ref_id]["curated"] = True
                changes["promoted"].append({
                    "id": ref_id,
                    "label": promo.get("label", refs[ref_id].get("label", "")),
                    "reason": promo.get("reason", "")
                })
        else:
            changes["errors"].append(f"Promotion target not found: {ref_id}")

    # Apply demotions
    for demote in aggregated["demotions"]:
        ref_id = demote["id"]
        if ref_id in refs:
            if refs[ref_id].get("curated", False):
                refs[ref_id]["curated"] = False
                changes["demoted"].append({
                    "id": ref_id,
                    "label": demote.get("label", refs[ref_id].get("label", "")),
                    "reason": demote.get("reason", "")
                })
        else:
            changes["errors"].append(f"Demotion target not found: {ref_id}")

    # Save updated citations
    with open(citations_path, "w") as f:
        json.dump(citations, f, indent=2)

    return changes

def main():
    print("Aggregating curation evaluations...")
    aggregated = aggregate_evaluations()

    print(f"\nAggregated decisions:")
    print(f"  Total evaluated: {aggregated['summary']['total_evaluated']}")
    print(f"  Promoted: {aggregated['summary']['promoted']}")
    print(f"  Demoted: {aggregated['summary']['demoted']}")
    print(f"  Kept curated: {aggregated['summary']['kept_curated']}")
    print(f"  Kept regular: {aggregated['summary']['kept_regular']}")

    print("\nApplying changes to citations.json...")
    changes = apply_curation_changes(aggregated)

    print(f"\nChanges applied:")
    print(f"  Promoted: {len(changes['promoted'])}")
    print(f"  Demoted: {len(changes['demoted'])}")
    print(f"  Errors: {len(changes['errors'])}")

    if changes["errors"]:
        print("\nErrors:")
        for error in changes["errors"]:
            print(f"  - {error}")

    # Save change log
    change_log_path = Path("viz/scans/curation_changes.json")
    with open(change_log_path, "w") as f:
        json.dump({
            "aggregated_summary": aggregated["summary"],
            "changes_applied": {
                "promoted_count": len(changes["promoted"]),
                "demoted_count": len(changes["demoted"]),
                "error_count": len(changes["errors"])
            },
            "promotions": changes["promoted"],
            "demotions": changes["demotions"],
            "errors": changes["errors"]
        }, f, indent=2)

    print(f"\nChange log saved to: {change_log_path}")

if __name__ == "__main__":
    main()
