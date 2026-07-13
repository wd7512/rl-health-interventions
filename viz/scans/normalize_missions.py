"""Normalize mission JSON files to a consistent schema."""

import json
import re
from pathlib import Path

SCANS_DIR = Path(__file__).parent


def extract_doi_from_url(url: str) -> str | None:
    if not url:
        return None
    m = re.search(r"10\.\d{4,}/[^\s]+", url)
    return m.group(0).rstrip(".") if m else None


def extract_arxiv_from_url(url: str) -> str | None:
    if not url:
        return None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
    if m:
        return f"arxiv:{m.group(1)}"
    m = re.search(r"arxiv:(\d+\.\d+)", url, re.IGNORECASE)
    if m:
        return f"arxiv:{m.group(1)}"
    return None


def normalize_paper(p: dict) -> dict:
    pid = p.get("id") or ""
    doi = p.get("doi") or ""
    arxiv_id = p.get("arxiv_id") or ""
    source_url = p.get("source_url") or p.get("url") or ""
    title = p.get("title") or p.get("label") or ""

    if not pid or pid == "null":
        if doi:
            pid = doi
        elif arxiv_id:
            pid = arxiv_id if arxiv_id.startswith("arxiv:") else f"arxiv:{arxiv_id}"
        elif source_url:
            extracted_doi = extract_doi_from_url(source_url)
            extracted_arxiv = extract_arxiv_from_url(source_url)
            if extracted_doi:
                pid = extracted_doi
            elif extracted_arxiv:
                pid = extracted_arxiv

    if not pid.startswith("10.") and not pid.startswith("arxiv:"):
        if source_url:
            extracted_doi = extract_doi_from_url(source_url)
            extracted_arxiv = extract_arxiv_from_url(source_url)
            if extracted_doi:
                pid = extracted_doi
            elif extracted_arxiv:
                pid = extracted_arxiv

    return {
        "id": pid,
        "label": title,
        "authors": p.get("authors", ""),
        "year": p.get("year"),
        "venue": p.get("venue") or p.get("journal") or "",
        "type": p.get("type", "Review"),
        "result": p.get("result") or p.get("positive_negative") or "null",
        "headline": p.get("headline") or p.get("key_finding") or p.get("summary") or "",
        "detail": p.get("detail") or p.get("abstract") or "",
        "sample_size": p.get("sample_size") or p.get("population") or "",
        "effect_size": p.get("effect_size") or "",
        "relevance": p.get("relevance", "medium"),
        "source_url": source_url or (f"https://doi.org/{pid}" if pid.startswith("10.") else ""),
        "notes": p.get("notes") or "",
    }


def main():
    for mf in sorted(SCANS_DIR.glob("mission_*.json")):
        with open(mf) as f:
            data = json.load(f)
        papers = data.get("papers", [])
        normalized = [normalize_paper(p) for p in papers]
        data["papers"] = normalized
        with open(mf, "w") as f:
            json.dump(data, f, indent=2)
        has_id = sum(1 for p in normalized if p["id"])
        print(f"{mf.name}: {len(normalized)} papers, {has_id} with IDs")


if __name__ == "__main__":
    main()
