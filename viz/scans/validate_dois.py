"""
DOI and arXiv ID validation script for research scan results.

Reads all mission_*.json files from viz/scans/, validates each paper's
identifier against CrossRef (DOIs) or arXiv API, and outputs a
validation_report.json with per-paper status.
"""

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SCANS_DIR = Path(__file__).parent
CROSSREF_API = "https://api.crossref.org/works/{}"
ARXIV_API = "https://export.arxiv.org/api/query?id_list={}"
RATE_LIMIT_DELAY = 1.0


def normalize_title(title: str) -> str:
    if not title:
        return ""
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def extract_first_author_last(authors_str: str) -> str:
    if not authors_str:
        return ""
    first = authors_str.split(",")[0].split(" and ")[0].split(" et al")[0].strip()
    parts = first.split()
    if not parts:
        return ""
    if len(parts) >= 2 and len(parts[-1]) <= 2 and parts[-1].isupper():
        return parts[0].lower()
    return parts[-1].lower()


def title_similarity(a: str, b: str) -> float:
    words_a = set(normalize_title(a).split())
    words_b = set(normalize_title(b).split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def validate_doi(doi: str, expected_title: str, expected_year: int, expected_authors: str) -> dict:
    url = CROSSREF_API.format(urllib.parse.quote(doi, safe=""))
    req = urllib.request.Request(url, headers={"User-Agent": "RLHealthScan/1.0 (mailto:research@scan.local)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"status": "not_found", "reason": "DOI not in CrossRef"}
        if e.code == 429:
            return {"status": "rate_limited", "reason": "CrossRef rate limit"}
        return {"status": "error", "reason": f"HTTP {e.code}"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    msg = data.get("message", {})
    actual_title = " ".join(msg.get("title", [""]))
    actual_years = msg.get("published-print", {}).get("date-parts", [[]])
    if not actual_years:
        actual_years = msg.get("published-online", {}).get("date-parts", [[]])
    actual_year = actual_years[0][0] if actual_years and actual_years[0] else None
    actual_authors = msg.get("author", [])
    first_author_family = actual_authors[0].get("family", "").lower() if actual_authors else ""

    issues = []
    sim = title_similarity(expected_title, actual_title)
    if sim < 0.5:
        issues.append(f"title mismatch (Jaccard={sim:.2f})")
    if expected_year and actual_year and abs(expected_year - actual_year) > 1:
        issues.append(f"year mismatch: expected {expected_year}, got {actual_year}")
    expected_last = extract_first_author_last(expected_authors)
    if expected_last and first_author_family and expected_last != first_author_family:
        issues.append(f"first author mismatch: expected '{expected_last}', got '{first_author_family}'")

    if issues:
        return {"status": "mismatch", "issues": issues, "actual_title": actual_title, "actual_year": actual_year}
    return {"status": "valid", "actual_title": actual_title, "actual_year": actual_year, "title_similarity": round(sim, 3)}


def validate_arxiv(arxiv_id: str, expected_title: str, expected_year: int, expected_authors: str) -> dict:
    clean_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
    url = ARXIV_API.format(urllib.parse.quote(clean_id, safe=""))
    req = urllib.request.Request(url, headers={"User-Agent": "RLHealthScan/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_text = resp.read().decode()
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    if "<entry>" not in xml_text and "<entry " not in xml_text:
        return {"status": "not_found", "reason": "arXiv ID not found"}

    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return {"status": "error", "reason": f"XML parse error: {e}"}

    entry = root.find("atom:entry", ns)
    if entry is None:
        return {"status": "not_found", "reason": "No entry in arXiv response"}

    actual_title = " ".join((entry.findtext("atom:title", "", ns) or "").split())
    published = entry.findtext("atom:published", "", ns) or ""
    actual_year = int(published[:4]) if len(published) >= 4 else None
    first_author = entry.find("atom:author/atom:name", ns)
    actual_first_author = first_author.text if first_author is not None else ""

    issues = []
    sim = title_similarity(expected_title, actual_title)
    if sim < 0.5:
        issues.append(f"title mismatch (Jaccard={sim:.2f})")
    if expected_year and actual_year and abs(expected_year - actual_year) > 1:
        issues.append(f"year mismatch: expected {expected_year}, got {actual_year}")
    expected_last = extract_first_author_last(expected_authors)
    actual_last = extract_first_author_last(actual_first_author)
    if expected_last and actual_last and expected_last != actual_last:
        issues.append(f"first author mismatch: expected '{expected_last}', got '{actual_last}'")

    if issues:
        return {"status": "mismatch", "issues": issues, "actual_title": actual_title, "actual_year": actual_year}
    return {"status": "valid", "actual_title": actual_title, "actual_year": actual_year, "title_similarity": round(sim, 3)}


def validate_paper(paper: dict) -> dict:
    pid = paper.get("id", "")
    title = paper.get("label", "") or paper.get("headline", "")
    year = paper.get("year")
    authors = paper.get("authors", "")
    if pid.startswith("10."):
        return validate_doi(pid, title, year, authors)
    elif pid.startswith("arxiv:"):
        return validate_arxiv(pid, title, year, authors)
    else:
        return {"status": "skipped", "reason": f"unrecognized ID format: {pid}"}


def main():
    mission_files = sorted(SCANS_DIR.glob("mission_*.json"))
    if not mission_files:
        print("No mission_*.json files found in viz/scans/", file=sys.stderr)
        sys.exit(1)

    report = {
        "scan_date": time.strftime("%Y-%m-%d"),
        "missions": {},
        "summary": {"total": 0, "valid": 0, "mismatch": 0, "not_found": 0, "error": 0, "skipped": 0, "rate_limited": 0},
    }

    for mf in mission_files:
        mission_name = mf.stem
        print(f"Validating {mission_name}...", file=sys.stderr)
        with open(mf) as f:
            data = json.load(f)
        papers = data.get("papers", [])
        mission_results = []
        for i, paper in enumerate(papers):
            result = validate_paper(paper)
            result["paper_id"] = paper.get("id", "")
            result["paper_label"] = paper.get("label", "")
            result["paper_year"] = paper.get("year")
            mission_results.append(result)
            report["summary"]["total"] += 1
            status = result.get("status", "error")
            if status in report["summary"]:
                report["summary"][status] += 1
            if i > 0 and i % 5 == 0:
                time.sleep(RATE_LIMIT_DELAY)
        report["missions"][mission_name] = mission_results

    output_path = SCANS_DIR / "validation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation complete. Report written to {output_path}", file=sys.stderr)
    s = report["summary"]
    print(f"  Total: {s['total']}  Valid: {s['valid']}  Mismatch: {s['mismatch']}  "
          f"Not found: {s['not_found']}  Error: {s['error']}  Skipped: {s['skipped']}", file=sys.stderr)


if __name__ == "__main__":
    main()
