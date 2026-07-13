"""Compute Pareto frontier for the citation graph.

Two dimensions:
  - Evidence: propagated from main paper base scores (empirical impact)
  - Definition relevance: text similarity to the strict definition

Pareto frontier = papers not dominated on both dimensions.
"""

import json
from pathlib import Path

import numpy as np
import requests

DATA_PATH = Path(__file__).parent / "citations.json"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text-v2-moe"

STRICT_DEFINITION = (
    "Reinforcement learning for personalizing just-in-time adaptive "
    "behavioral interventions (JITAIs) in mobile health, learning treatment "
    "policies from micro-randomized trial data under constraints of "
    "non-stationarity, habituation, and user burden."
)

# Base evidence: 3=RCT efficacy, 2=deployed w/ outcomes, 1=indirect, 0=dataset/tool
BASE_EVIDENCE = {
    "10.1007/s12160-018-0049-1": 3,  # HeartSteps V1 — MRT, PA efficacy
    "arxiv:1909.03539": 3,  # HeartSteps V2 — MRT, PA efficacy
    "arxiv:2508.10060": 1,  # PEARL — off-policy eval (indirect)
    "arxiv:2411.00336": 2,  # StepCountJITAI — deployed JITAI
    "10.1038/s41597-023-02763-w": 0,  # Health Gym — synthetic RL env
    "10.1001/jamacardio.2016.4395": 1,  # MyHeartCounts — observational
    "10.1038/s41746-024-01062-3": 0,  # UK Biobank — dataset
    "10.1038/s41591-026-04352-3": 0,  # All of Us — dataset
    "10.13026/5M0E-GX79": 0,  # NHANES — dataset
    "10.1037/0033-2909.136.6.969": 0,  # COM-B Framework — theory
}

data = json.loads(DATA_PATH.read_text())
papers: dict = data["papers"]
refs: dict = data["refs"]


def build_text(entry: dict) -> str:
    parts = [entry.get("label", ""), entry.get("headline", ""), entry.get("detail", "")]
    return " | ".join(p for p in parts if p)


def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=30
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


EPS = 1e-10

def normalize(vals: list[float]) -> list[float]:
    mn, mx = min(vals), max(vals)
    if mx - mn < EPS:
        return [0.5] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]


# Embed definition
print("Embedding strict definition...")
def_emb = np.array(get_embedding(STRICT_DEFINITION), dtype=np.float32)

# Embed all entries
all_entries = [(pid, p, "paper") for pid, p in papers.items()] + [
    (rid, r, "ref") for rid, r in refs.items()
]
print(f"Computing {len(all_entries)} embeddings...")
embed_cache = {}
for i, (_eid, entry, _kind) in enumerate(all_entries):
    txt = build_text(entry)
    if txt not in embed_cache:
        embed_cache[txt] = np.array(get_embedding(txt), dtype=np.float32)
    entry["_def_emb"] = embed_cache[txt]
    if (i + 1) % 40 == 0:
        print(f"  {i + 1}/{len(all_entries)}")

# Step 1: compute def_rel for papers first (need it for weighted propagation)
paper_def_rel = {}
for pid, p in papers.items():
    paper_def_rel[pid] = cos_sim(p["_def_emb"], def_emb)

# Step 2: continuous evidence scores
# For papers: base_evidence + small def_rel contribution to break ties
# For refs: sum over citing papers of (base + 0.1*def_rel) — citing papers
# that are both high-evidence AND high-relevance pass more evidence downstream
evidence = {}
for pid in papers:
    evidence[pid] = float(BASE_EVIDENCE.get(pid, 0)) + 0.1 * paper_def_rel[pid]
for rid in refs:
    citing = [pid for pid in papers if rid in papers[pid]["cites"]]
    evidence[rid] = sum(
        float(BASE_EVIDENCE.get(pid, 0)) + 0.1 * paper_def_rel[pid] for pid in citing
    )

# Step 3: compute definition relevance for all entries
def_rel = {}
for eid, entry, _ in all_entries:
    def_rel[eid] = cos_sim(entry["_def_emb"], def_emb)
    entry.pop("_def_emb", None)

# Collect scores for all entries
all_scores = []
for eid, entry, kind in all_entries:
    all_scores.append(
        {
            "id": eid,
            "label": entry.get("label", eid),
            "kind": kind,
            "evidence": evidence[eid],
            "def_rel": def_rel[eid],
        }
    )

ev_norm = normalize([s["evidence"] for s in all_scores])
dr_norm = normalize([s["def_rel"] for s in all_scores])
for i, s in enumerate(all_scores):
    s["ev_norm"] = ev_norm[i]
    s["dr_norm"] = dr_norm[i]

# Strict Pareto frontier: non-dominated set
frontier_ids = set()
for a in all_scores:
    dominated = False
    for b in all_scores:
        if b["id"] == a["id"]:
            continue
        if (
            b["ev_norm"] >= a["ev_norm"]
            and b["dr_norm"] >= a["dr_norm"]
            and (b["ev_norm"] > a["ev_norm"] or b["dr_norm"] > a["dr_norm"])
        ):
            dominated = True
            break
    if not dominated:
        frontier_ids.add(a["id"])

# Composite frontier_score = distance from origin weighting evidence more
for s in all_scores:
    s["frontier_score"] = s["ev_norm"] * 0.6 + s["dr_norm"] * 0.4

# Report
f_refs = [
    s for s in all_scores if s["id"].startswith("ref:") and s["id"] in frontier_ids
]
f_papers = [
    s for s in all_scores if not s["id"].startswith("ref:") and s["id"] in frontier_ids
]
print(f"\nPareto frontier: {len(frontier_ids)} of {len(all_scores)} entries")
print(f"  Frontier papers: {len(f_papers)}")
for s in sorted(f_papers, key=lambda x: -x["ev_norm"]):
    print(f"    {s['label']}  ev={s['ev_norm']:.3f} rel={s['dr_norm']:.3f}")
print(f"  Frontier refs: {len(f_refs)}")
for s in sorted(f_refs, key=lambda x: -x["frontier_score"]):
    print(f"    {s['label']}  ev={s['ev_norm']:.3f} rel={s['dr_norm']:.3f}")
print("\nTop 25 by frontier_score:")
for s in sorted(all_scores, key=lambda x: -x["frontier_score"])[:25]:
    m = "★" if s["id"] in frontier_ids else "·"
    print(f"  {m} {s['frontier_score']:.4f}  {s['label']} ({s['kind']})")

# Build score lookup for write-back
score_map = {s["id"]: s for s in all_scores}
for eid, entry, _ in all_entries:
    entry["frontier"] = eid in frontier_ids
    entry["evidence_score"] = round(evidence[eid], 4)
    entry["def_rel"] = round(def_rel[eid], 4)
    entry["frontier_score"] = round(score_map[eid]["frontier_score"], 4)

DATA_PATH.write_text(json.dumps(data, indent=2))
print(f"\nWritten frontier flags to {DATA_PATH}")
