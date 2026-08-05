"""Score all papers/refs relative to HeartSteps V2 for interactive curation."""

import json
from pathlib import Path

import numpy as np
import requests

DATA_PATH = Path(__file__).parent / "citations.json"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text-v2-moe"
HSV2_ID = "arxiv:1909.03539"
PEARL_ID = "arxiv:2508.10060"

data = json.loads(DATA_PATH.read_text())
papers: dict = data["papers"]
refs: dict = data["refs"]

hsv2 = papers[HSV2_ID]
hsv2_year = hsv2["year"]
hsv2_refs = {c for c in hsv2["cites"] if c.startswith("ref:")}
hsv2_paper_cites = {c for c in hsv2["cites"] if not c.startswith("ref:")}


def build_text(entry: dict) -> str:
    parts = [entry.get("label", ""), entry.get("headline", ""), entry.get("detail", "")]
    return " | ".join(p for p in parts if p)


def get_embedding(text: str) -> list[float]:
    resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ── Build ordered lists ──────────────────────────────────────────
embed_cache: dict[str, np.ndarray] = {}
all_main_entries: list[tuple[str, dict]] = [(pid, p) for pid, p in papers.items()]
all_ref_entries: list[tuple[str, dict]] = [(rid, r) for rid, r in refs.items()]

total = len(all_main_entries) + len(all_ref_entries)
print(f"Computing {total} embeddings via Ollama ({EMBED_MODEL})...")
for label, entries in [("main papers", all_main_entries), ("refs", all_ref_entries)]:
    for i, (eid, entry) in enumerate(entries):
        txt = build_text(entry)
        if txt not in embed_cache:
            embed_cache[txt] = np.array(get_embedding(txt), dtype=np.float32)
        entry["_emb"] = embed_cache[txt]
        if (i + 1) % 20 == 0:
            print(f"  {label}: {i + 1}/{len(entries)}")

hsv2_emb = embed_cache.get(build_text(hsv2), np.zeros(768))

# ── Compute signals for main papers ──────────────────────────────

main_paper_info: list[dict] = []
for pid, p in all_main_entries:
    p_refs = {c for c in p["cites"] if c.startswith("ref:")}
    p_paper_cites = {c for c in p["cites"] if not c.startswith("ref:")}
    shared = p_refs & hsv2_refs
    all_union = p_refs | hsv2_refs
    shared_pct = len(shared) / len(all_union) if all_union else 0
    text_sim = cos_sim(p["_emb"], hsv2_emb)
    cites_hsv2 = HSV2_ID in p["cites"]
    cited_by_hsv2 = pid in hsv2_paper_cites
    direct = 1 if cites_hsv2 or cited_by_hsv2 else 0
    year_gap = 1 - abs(p["year"] - hsv2_year) / (max(p["year"], hsv2_year) - 1900 + 1)

    # Co-citation: other main papers that cite BOTH this paper AND HS-V2
    co_cited = 0
    for other_pid, other_p in papers.items():
        if other_pid == pid or other_pid == HSV2_ID:
            continue
        if pid in other_p["cites"] and HSV2_ID in other_p["cites"]:
            co_cited += 1

    main_paper_info.append({
        "id": pid,
        "label": p["label"],
        "year": p["year"],
        "authors": p.get("authors", ""),
        "shared_refs": len(shared),
        "shared_total": len(all_union),
        "shared_pct": shared_pct,
        "text_sim": text_sim,
        "cites_hsv2": cites_hsv2,
        "cited_by_hsv2": cited_by_hsv2,
        "direct": direct,
        "co_cited": co_cited,
        "year_gap": year_gap,
        "venue": p.get("venue", ""),
        "headline": p.get("headline", ""),
        "detail": p.get("detail", ""),
    })

# ── Compute signals for refs ─────────────────────────────────────

ref_info: list[dict] = []
for rid, r in all_ref_entries:
    citing_papers = [pid for pid, p in papers.items() if rid in p["cites"]]
    num_citing = len(citing_papers)
    cited_by_hsv2 = HSV2_ID in citing_papers
    text_sim = cos_sim(r["_emb"], hsv2_emb)
    year_gap = 1 - abs(r["year"] - hsv2_year) / (max(r["year"], hsv2_year) - 1900 + 1) if r.get("year") else 0

    ref_info.append({
        "id": rid,
        "label": r["label"],
        "year": r.get("year", 0),
        "authors": r.get("authors", ""),
        "cited_by_hsv2": cited_by_hsv2,
        "num_citing": num_citing,
        "text_sim": text_sim,
        "year_gap": year_gap,
        "curated": r.get("curated", False),
        "venue": r.get("venue", ""),
        "headline": r.get("headline", ""),
        "detail": r.get("detail", ""),
        "citing_papers": citing_papers,
    })


def norm(vals: list[float]) -> list[float]:
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-10:
        return [0.5] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]


# ── Score & rank main papers ─────────────────────────────────────

W_SHARED, W_TEXT, W_DIRECT, W_COCITED, W_YEAR = 0.35, 0.20, 0.20, 0.15, 0.10

s_shared = norm([m["shared_pct"] for m in main_paper_info])
s_text_m = norm([m["text_sim"] for m in main_paper_info])
s_direct = norm([float(m["direct"]) for m in main_paper_info])
s_cocited = norm([float(m["co_cited"]) for m in main_paper_info])
s_year_m = norm([m["year_gap"] for m in main_paper_info])

for i, m in enumerate(main_paper_info):
    m["score"] = (
        W_SHARED * s_shared[i]
        + W_TEXT * s_text_m[i]
        + W_DIRECT * s_direct[i]
        + W_COCITED * s_cocited[i]
        + W_YEAR * s_year_m[i]
    )

main_paper_info.sort(key=lambda m: -m["score"])

# ── Score & rank refs ────────────────────────────────────────────

W_CITEDBY, W_TEXT_R, W_MULTI, W_YEAR_R = 0.40, 0.30, 0.20, 0.10

s_cited_by = norm([float(r["cited_by_hsv2"]) for r in ref_info])
s_text_r = norm([r["text_sim"] for r in ref_info])
s_multi = norm([float(r["num_citing"]) for r in ref_info])
s_year_r = norm([r["year_gap"] for r in ref_info])

for i, r in enumerate(ref_info):
    r["score"] = (
        W_CITEDBY * s_cited_by[i]
        + W_TEXT_R * s_text_r[i]
        + W_MULTI * s_multi[i]
        + W_YEAR_R * s_year_r[i]
    )

ref_info.sort(key=lambda r: -r["score"])


def paper_label(pid: str) -> str:
    p = papers.get(pid)
    return p["label"] if p else refs.get(pid, {}).get("label", pid)


SEP = "─" * 120
print(f"\n{SEP}")
print("MAIN PAPERS — ranked by composite connection to HeartSteps V2")
print(f"      score = {W_SHARED:.0%} shared_refs + {W_TEXT:.0%} text_sim + {W_DIRECT:.0%} direct_cite + {W_COCITED:.0%} co_cited + {W_YEAR:.0%} year_prox")
print(SEP)
print(f"{'#':<4} {'Score':<6} {'Shared':<20} {'TSim':<5} {'Dir':<4} {'CoCt':<5} {'Year':<5}  Paper")
print(SEP)

for i, m in enumerate(main_paper_info):
    sstr = f"{m['shared_refs']}/{m['shared_total']}" if m['shared_total'] > 0 else "0/0"
    dstr = "⇄" if m["cites_hsv2"] and m["cited_by_hsv2"] else "←" if m["cites_hsv2"] else "→" if m["cited_by_hsv2"] else "—"
    pearl = " ★" if m["id"] == PEARL_ID else ""
    print(f"{i+1:<4} {m['score']:.3f} {sstr:<20} {m['text_sim']:.3f} {dstr:<4} {m['co_cited']:<5} {m['year']:<5}  {m['label']}{pearl}")

print(f"\n{SEP}")
print("REFS — ranked by composite connection to HeartSteps V2")
print(f"      score = {W_CITEDBY:.0%} cited_by_hsv2 + {W_TEXT_R:.0%} text_sim + {W_MULTI:.0%} multi_cite + {W_YEAR_R:.0%} year_prox")
print(SEP)
print(f"{'#':<4} {'Score':<6} {'HSV2':<5} {'Citers':<20} {'TSim':<5} {'Year':<5} {'Cur?':<6}  Paper")
print(SEP)

for i, r in enumerate(ref_info):
    hsv2m = "✓" if r["cited_by_hsv2"] else "·"
    citer_labels = [paper_label(c) for c in sorted(r["citing_papers"])]
    citer_disp = f"{r['num_citing']}→" + ", ".join(citer_labels[:2])
    if len(citer_labels) > 2:
        citer_disp += ",…"
    cur = "KEPT" if r["curated"] and r["cited_by_hsv2"] else "DEMOTE" if r["curated"] else "PROMOTE" if r["cited_by_hsv2"] else ""
    print(f"{i+1:<4} {r['score']:.3f} {hsv2m:<5} {citer_disp:<20} {r['text_sim']:.3f} {r['year']:<5} {cur:<6}  {r['label']}")

print(f"\nTotal: {len(main_paper_info)} main papers, {len(ref_info)} refs")
print(f"Embeddings via {EMBED_MODEL} on local Ollama")

