---
name: research-landscape-scan
description: Use when expanding the citation graph with new papers from the research landscape. Launches 10 parallel subagents to scan literature, validates DOIs, merges results into citations.json, and updates the visualization dashboard.
---

# Research Landscape Scan

Expand the citation graph by systematically scanning the research landscape for papers relevant to RL for health interventions. Uses 10 parallel subagents to cover different research domains, validates all discovered papers against CrossRef/arXiv APIs, and merges validated results into the visualization dataset.

## Prerequisites

- `viz/citations.json` exists with current dataset
- `viz/scans/` directory (will be created if missing)
- Python 3.10+ with `requests` library
- Internet access for API validation

## Workflow

### 1. Define Mission Scope

Identify 10 research domains to scan. Each mission should:
- Cover a distinct aspect of the research landscape
- Include both positive and negative results
- Focus on papers from 2020-2026 (or user-specified range)
- Target 15-25 papers per mission

**Completion criterion:** 10 mission topics defined with clear search terms and inclusion criteria.

### 2. Launch Parallel Subagents

Launch 10 `explore` subagents in parallel. Each subagent receives:

**Shared context block:**
```
PROJECT CONTEXT:
- Config-driven RL for Just-in-Time Adaptive Interventions (JITAIs)
- 36-state factored MDP, 4 actions, 5 behavioral personas
- HeartSteps V1/V2, PEARL, StepCountJITAI already covered
- Next steps: sensor FMs, more LLMs, synthetic clinical results

SCOPE: {YEAR_RANGE} only. Include negative results and failed trials.
No venue bias. Mark arXiv preprints in notes.
```

**Mission-specific prompt:**
- Topic name and mission number
- Key search terms (4-6 queries)
- Domains to cover
- Negative result emphasis

**Output requirements:**
Each subagent writes TWO files to `viz/scans/`:
1. `mission_{NN}_{topic}.json` — structured paper data
2. `mission_{NN}_{topic}_report.md` — narrative summary

**JSON schema:**
```json
{
  "mission_id": "{NN}",
  "topic": "Topic Name",
  "scan_date": "YYYY-MM-DD",
  "papers": [
    {
      "id": "DOI or arXiv ID",
      "label": "short title",
      "authors": "First et al.",
      "year": 2024,
      "venue": "Conference/Journal",
      "type": "Trial|Dataset|Framework|Simulator|Benchmark|Review",
      "result": "positive|negative|mixed|null",
      "headline": "one-line result",
      "detail": "2-3 sentence description",
      "sample_size": "n=X or null",
      "effect_size": "quantitative result or null",
      "relevance": "high|medium|low",
      "source_url": "https://doi.org/... or https://arxiv.org/...",
      "notes": "caveats, limitations"
    }
  ],
  "summary": {
    "total_papers": 15,
    "positive_results": 10,
    "negative_results": 3,
    "mixed_results": 2,
    "key_findings": ["finding 1", "finding 2", "finding 3"]
  }
}
```

**Completion criterion:** All 10 subagents complete, 20 files written to `viz/scans/`.

### 3. Normalize and Validate

Run validation pipeline:

```bash
# Normalize JSON schemas across missions
uv run viz/scans/normalize_missions.py

# Validate DOIs against CrossRef, arXiv IDs against arXiv API
uv run viz/scans/validate_dois.py
```

**Validation checks:**
- DOI reachable via CrossRef API
- Title match (Jaccard similarity ≥ 0.5)
- Year match (±1 tolerance)
- First author last name match
- arXiv ID reachable via arXiv API

**Expected output:**
- `viz/scans/validation_report.json` with per-paper status
- Statuses: `valid`, `mismatch`, `not_found`, `error`, `rate_limited`, `skipped`

**Completion criterion:** Validation report generated. Review mismatches manually if >20% of papers.

### 4. Aggregate Results

Merge all mission results into single dataset:

```bash
uv run viz/scans/aggregate.py
```

**Output:**
- `viz/scans/aggregated.json` — all papers deduplicated by ID
- `viz/scans/AGGREGATE_REPORT.md` — summary with top recommendations

**Completion criterion:** Aggregated dataset created with deduplication count reported.

### 5. Merge to Citations

Add validated papers to main dataset:

```bash
# Backup current dataset
cp viz/citations.json viz/citations.json.backup

# Merge scan papers as refs
uv run viz/scans/merge_to_citations.py
```

**Merge logic:**
- Each scan paper becomes a ref with ID `ref:scan:{NNN}`
- Preserves source DOI/arXiv ID in `source_id` field
- Tracks result type in `scan_result` field
- Tracks mission origins in `missions` field
- Prefixes negative results with `[Negative]` in headlines
- Sets `curated: true` for Trial/Dataset/Framework/Simulator/Benchmark types

**Completion criterion:** `viz/citations.json` expanded, no schema violations.

### 6. Full Dataset Validation

Validate entire dataset integrity:

```bash
uv run viz/validate_full_dataset.py
```

**Checks:**
- All required fields present (label, authors, year, venue, type, cites)
- All citation targets exist
- DOI/arXiv validation for main papers
- JSON structure valid

**Output:**
- `viz/validation_full_report.json` with issue list
- Exit code 0 if no issues, 1 if issues found

**Completion criterion:** All structural checks pass. DOI mismatches in original main papers are acceptable (pre-existing).

### 7. Update Visualization

The viz app automatically picks up new papers. To see changes:

1. Hard refresh browser (Cmd+Shift+R / Ctrl+Shift+R)
2. Switch to "Full" mode to see all refs
3. New scan papers appear as:
   - Dark dots (non-curated refs)
   - Labeled dots (curated refs: Trial/Dataset/Framework/Simulator/Benchmark)

**Optional: Recompute evidence scores**

```bash
# Requires Ollama running with nomic-embed-text-v2-moe
uv run viz/curate.py
```

This recalculates `evidence_score`, `def_rel`, and `frontier_score` for all refs based on connection to HeartSteps V2.

**Completion criterion:** Viz app loads without errors, new papers visible in Full mode.

## Output Structure

```
viz/scans/
├── mission_01_rl_pa.json
├── mission_01_rl_pa_report.md
├── ...
├── mission_10_platforms.json
├── mission_10_platforms_report.md
├── normalize_missions.py
├── validate_dois.py
├── aggregate.py
├── merge_to_citations.py
├── validation_report.json
├── aggregated.json
└── AGGREGATE_REPORT.md

viz/
├── citations.json (expanded)
├── citations.json.backup (original)
├── validate_full_dataset.py
└── validation_full_report.json
```

## Failure Modes

- **Subagent rate limiting:** Some missions may hit arXiv/PubMed rate limits. Retry after 5 minutes or reduce parallelism.
- **Schema mismatches:** Different subagents may use different field names. `normalize_missions.py` handles common variations.
- **DOI not found:** Some DOIs may not be indexed in CrossRef yet (especially recent papers). Mark as `not_found` but keep in dataset.
- **Title mismatches:** Short labels vs full titles cause Jaccard similarity failures. Review manually if >20% mismatch rate.

## Scaling

- **More missions:** Add missions 11-20 by duplicating mission template and updating IDs.
- **Different domains:** Adapt search terms and context block for non-RL-health topics.
- **Historical scans:** Change year range in scope (e.g., 2010-2020 for foundational work).

## References

- Scripts: `viz/scans/*.py`
- Data: `viz/citations.json`, `viz/scans/aggregated.json`
- Viz app: `viz/index.html`
- Existing skills: `.opencode/skills/dev-workflow/SKILL.md`
