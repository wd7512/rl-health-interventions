"""Validate the entire citations.json dataset."""

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CITATIONS_PATH = Path(__file__).parent / "citations.json"
CROSSREF_API = "https://api.crossref.org/works/{}"
ARXIV_API = "https://export.arxiv.org/api/query?id_list={}"


def normalize_title(title: str) -> str:
    if not title:
        return ""
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def title_similarity(a: str, b: str) -> float:
    words_a = set(normalize_title(a).split())
    words_b = set(normalize_title(b).split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def validate_doi(doi: str, expected_title: str, expected_year: int) -> dict:
    url = CROSSREF_API.format(urllib.parse.quote(doi, safe=""))
    req = urllib.request.Request(url, headers={"User-Agent": "CitationsValidator/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"status": "not_found"}
        if e.code == 429:
            return {"status": "rate_limited"}
        return {"status": "error", "reason": f"HTTP {e.code}"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    msg = data.get("message", {})
    actual_title = " ".join(msg.get("title", [""]))
    actual_years = msg.get("published-print", {}).get("date-parts", [[]])
    if not actual_years:
        actual_years = msg.get("published-online", {}).get("date-parts", [[]])
    actual_year = actual_years[0][0] if actual_years and actual_years[0] else None

    sim = title_similarity(expected_title, actual_title)
    if sim < 0.4:
        return {"status": "mismatch", "similarity": round(sim, 3)}
    if expected_year and actual_year and abs(expected_year - actual_year) > 1:
        return {"status": "year_mismatch", "expected": expected_year, "actual": actual_year}
    return {"status": "valid", "similarity": round(sim, 3)}


def validate_arxiv(arxiv_id: str, expected_title: str, expected_year: int) -> dict:
    clean_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
    url = ARXIV_API.format(urllib.parse.quote(clean_id, safe=""))
    req = urllib.request.Request(url, headers={"User-Agent": "CitationsValidator/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_text = resp.read().decode()
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    if "<entry>" not in xml_text and "<entry " not in xml_text:
        return {"status": "not_found"}

    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return {"status": "error", "reason": f"XML parse error: {e}"}

    entry = root.find("atom:entry", ns)
    if entry is None:
        return {"status": "not_found"}

    actual_title = " ".join((entry.findtext("atom:title", "", ns) or "").split())
    published = entry.findtext("atom:published", "", ns) or ""
    actual_year = int(published[:4]) if len(published) >= 4 else None

    sim = title_similarity(expected_title, actual_title)
    if sim < 0.4:
        return {"status": "mismatch", "similarity": round(sim, 3)}
    if expected_year and actual_year and abs(expected_year - actual_year) > 1:
        return {"status": "year_mismatch", "expected": expected_year, "actual": actual_year}
    return {"status": "valid", "similarity": round(sim, 3)}


def validate_dataset():
    with open(CITATIONS_PATH) as f:
        data = json.load(f)

    papers = data.get("papers", {})
    refs = data.get("refs", {})
    all_ids = set(papers.keys()) | set(refs.keys())

    print(f"Validating citations.json...")
    print(f"  Main papers: {len(papers)}")
    print(f"  Refs: {len(refs)}")
    print(f"  Total IDs: {len(all_ids)}")
    print()

    issues = []
    citation_targets = set()
    doi_validations = {"valid": 0, "mismatch": 0, "not_found": 0, "error": 0, "rate_limited": 0, "skipped": 0}

    for section_name, section in [("papers", papers), ("refs", refs)]:
        for item_id, item in section.items():
            if not item.get("label"):
                issues.append(f"{item_id}: missing label")
            if not item.get("authors"):
                issues.append(f"{item_id}: missing authors")
            if not item.get("year"):
                issues.append(f"{item_id}: missing year")

            for cite_id in item.get("cites", []):
                citation_targets.add(cite_id)
                if cite_id not in all_ids:
                    issues.append(f"{item_id} cites non-existent ID: {cite_id}")

            if item_id.startswith("10."):
                result = validate_doi(item_id, item.get("label", ""), item.get("year"))
                status = result["status"]
                if status == "valid":
                    doi_validations["valid"] += 1
                elif status in ("mismatch", "year_mismatch"):
                    doi_validations["mismatch"] += 1
                    issues.append(f"{item_id}: {status}")
                elif status == "not_found":
                    doi_validations["not_found"] += 1
                    issues.append(f"{item_id}: not found in CrossRef")
                elif status == "rate_limited":
                    doi_validations["rate_limited"] += 1
                else:
                    doi_validations["error"] += 1
                time.sleep(0.5)
            elif item_id.startswith("arxiv:"):
                result = validate_arxiv(item_id, item.get("label", ""), item.get("year"))
                status = result["status"]
                if status == "valid":
                    doi_validations["valid"] += 1
                elif status in ("mismatch", "year_mismatch"):
                    doi_validations["mismatch"] += 1
                    issues.append(f"{item_id}: {status}")
                elif status == "not_found":
                    doi_validations["not_found"] += 1
                    issues.append(f"{item_id}: not found in arXiv")
                else:
                    doi_validations["error"] += 1
                time.sleep(0.5)
            else:
                doi_validations["skipped"] += 1

    print("DOI/arXiv validation:")
    for k, v in doi_validations.items():
        print(f"  {k}: {v}")
    print()

    if issues:
        print(f"Issues found: {len(issues)}")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("No issues found!")

    report = {
        "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_papers": len(papers),
        "total_refs": len(refs),
        "total_ids": len(all_ids),
        "citation_targets": len(citation_targets),
        "doi_validations": doi_validations,
        "issues": issues,
        "issue_count": len(issues),
    }

    report_path = CITATIONS_PATH.parent / "validation_full_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {report_path}")

    return len(issues) == 0


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
