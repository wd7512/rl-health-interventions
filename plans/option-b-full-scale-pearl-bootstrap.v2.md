# Option B Completion — Full-Scale PEARL Bootstrap (FINALIZED)

**Date:** 2026-08-01
**Branch:** `feat/full-scale-pearl-bootstrap` (from main `f847c60`)
**Status:** Approved — ready to execute
**Related:** #288 (Option B hardening, closed by #289); future PRs for pilot + full-scale

---

## 1. Executive Summary

PR #289 (Option B hardening, rounds 11-13) is **merged into main as `f847c60`**;
`protocol_fewshot` r13 is FROZEN (mean +223.2 in the +150-450 band, min −30.0,
47/48 positive, C3 green, 0 parse failures). This plan: (1) finishes the pilot
experiments that were already underway — a temperature sweep and an n=6
convergence pilot (~624 calls), (2) generates the full 108-state × 13-action
bootstrap table (1,404 cells × 10 samples = **14,040 calls**, user-approved),
(3) applies the pending constitution Layer-2 fixes and validates all 4 tiers on
the full table, (4) ships.

**Locked decisions (user-confirmed):**
1. **Proceed with 10 samples/cell** (14,040 calls, ~2-3 h at 50 workers,
   est. low tens of dollars).
2. **Temperature decided by pilot** — temp sweep (0.3 vs 0.7 at n=3), then
   n=6 convergence pilot on the winning temp.
3. **Pilot sizing: n=3 temp sweep + n=6 convergence** (~624 calls total).
4. **Aggregator extracted** to `src/rl_health_interventions/llm_bootstrapping/
   table_aggregate.py` with a sum-to-1 correctness fix + unit tests.
5. **Full table named `tables/pearl_12action/pearl_bootstrap.json`; stale
   `pearl_random.json` deleted** (burden `low/medium/high` is incompatible with
   the config's `none/minor/major`; the table loader globs all `*.json` in
   `table_dir`).

---

## 2. Current branch state

- Branch `feat/full-scale-pearl-bootstrap` created from main `f847c60`.
- **Uncommitted working tree:**
  - `scripts/pearl_recalibration/generate_pearl_mini.py` — PARTIAL edit adding
    `--temperature` (argparse + first `batch_complete` wired; the retry
    `batch_complete` call is NOT yet wired). Completing this is step 1 of
    Phase 1.
  - `plans/option-b-full-scale-pearl-bootstrap.v1.md` — draft plan (untracked).
- Main = `f847c60` (PR #289 merged: literature doc, prompt rounds 11-13 freeze,
  robust analyzer, bounded retry with original-preservation, log/archives).

---

## 3. Phase 1 — Finish pilot experiments (~624 calls, decision-gated)

### 3.1 Complete the `--temperature` flag (code)
- Finish the partial edit in `generate_pearl_mini.py`: pass `temperature` to the
  **retry** `batch_complete` call as well (default 0.7 when not given).
- Unit test for the flag plumbing (argparse default + passthrough).

### 3.2 Temperature sweep (n=3 × 2 temps = 312 calls)
- Archive the r13 table first (`archive/pearl_pilot_protocol_fewshot_r13.json`).
- Run frozen `protocol_fewshot` at **0.3** and **0.7** on the 4-state mini set
  (156 calls each).
- Compare: mean/median/trimmed lift, min/max cell lift, per-cell variance,
  positives count, parse failures, C3.
- Winner: best ceiling behavior (lowest max-cell overshoot) while keeping mean
  in the +150-450 band and ≥47/48 positive.

### 3.3 Convergence pilot (n=6, 312 calls)
- Winner temp at 6 samples × 52 cells on the same 4 states.
- Question: does the r13 max-cell overshoot (+823.8) persist at n=6 (real
  signal) or shrink (n=3 noise)? Do per-cell means stabilize vs n=3?
- This is the evidence gate supporting the 10-sample full run.

### 3.4 Decision + logging
- Record rounds 14 (temp sweep) and 15 (convergence) in
  `docs/research/prompt-refinement-log.{json,md}`.
- If convergence is poor or overshoot is structural, surface to the user before
  the 14,040-call spend.

---

## 4. Phase 2 — Full-scale generation (14,040 calls, approved)

### 4.1 Extract the aggregator (code, testable)
- Move `_aggregate_to_table` from `generate_pearl_mini.py` to a new package
  module `src/rl_health_interventions/llm_bootstrapping/table_aggregate.py`;
  both generator scripts import it.
- **Correctness fix (pre-flight finding):** `round(v/total, 4)` can produce
  sums like 0.9999 (e.g., 1/1/1 split at n=3, 2/2/2 at n=6), and
  `TableValidator._check_probability_sum` requires |sum−1.0| ≤ 1e-6
  (`_table_validator.py:11,145-151`). The pilot table passed only because no
  cell hit such a split. Fix: round all but the last probability and set
  `last = 1 − sum(others)`.
- Unit tests: 3-level `recent_steps_mean` (low/moderate/high — pilot only
  exercised 2 levels), sum-to-1 invariant, `_MIN_SAMPLES_PER_CELL` drop,
  unknown-factor tolerance.

### 4.2 New `scripts/pearl_recalibration/generate_pearl_full.py`
- 108 × 13 = 1,404 cells × 10 samples = 14,040 calls, frozen
  `protocol_fewshot` at temperature 0.3.
- **Design (rework, mirrors Sprint 1 `request.py` + `request_helper.py`):**
  - **Cell-level resume**, not state-chunk buffering. A cell = one
    (state, action) needing `--samples` responses. `_todo_cells` counts
    `content` records already on disk (errors do not count) and tops up only
    the shortfall per cell.
  - **Per-batch append**: `_append_raw` runs after every `batch_complete`
    call (default `--batch-size 100` prompts), so a stalled request never
    risks losing completed work — everything on disk survives and `--resume`
    re-runs only what is missing.
  - `--resume` tops up incomplete cells; `--retry-errors` first strips error
    records (`_strip_errors`) then tops up; `--finalize-only` aggregates the
    raw file with no API calls.
  - Prompts are rendered per cell via `_render_user_prompt` (exact shortfall
    counts, no state-chunk grouping); bounded retry on unparseable responses
    with `original_content`/`original_error` preservation.
- Writes `tables/pearl_12action/pearl_bootstrap.json` (burden
  `none/minor/major`, matching the config vocabulary).
- Pace: litellm `future.result()` has no timeout, so one slow request can add
  ~13 min per batch (per-request 600s timeout + 7 retries). Per-batch append +
  resume make this survivable rather than fatal.
- Cost est. low tens of dollars (deepseek-v4-flash, OpenRouter).

### 4.3 Validation
- `TableValidator` against `config/pearl_constitution_12action.yaml` (all 3
  stochastic factors per cell, sums to 1).
- Coverage check: 1,404 unique (state, action), all 13 actions × 108 states.
- Simulator smoke test reusing `validate_pearl_mini.py`'s temp-dir isolation
  (the loader globs ALL `*.json` in `table_dir`, so isolation is mandatory).
- **Delete stale `tables/pearl_12action/pearl_random.json`** — it must not sit
  beside the new table.

### 4.4 Pipeline-level ceiling handling (r13 freeze decision)
- Lift analysis uses median/trimmed robust metrics (already in the analyzer).
- Transitions are bin-normalized, so residual high-side overshoot in raw
  responses cannot distort the table itself; document any per-cell caps applied
  in analysis.

---

## 5. Phase 3 — Constitution Layer-2 fixes + validation

### 5.1 Persist the gap map
- `docs/research/constitution-gaps.md` — currently the gap map lives only in
  conversation; persist it first so the fixes are grounded.

### 5.2 Fixes (verify each against the persisted gap map)
1. **T2.3 ordering** (`run_distribution_check.py:86`): strict
   `RL ≥ Fixed ≥ Random > Control` → `RL > (Fixed ≈ Random ≈ Control)` per gap
   map.
2. **T2.2 band** (`run_distribution_check.py:58`): gap map says 210-296 band;
   current check uses 150-450. **Verify the intended band before changing** —
   resolve against the PEARL reference and r13 ladder results.
3. **μ 5,618 vs 5,580**: resolve the discrepancy between `pearl_reference.json`
   (`mean_baseline_steps: 5618.2`) and the 5,580 figure in `utils.py`
   `_RECENT_STEPS_MEAN_MIDPOINT` comment; make check constants derive from
   `load_reference()` where possible.
4. **persona_name wiring**: `load_constitution_config` (utils.py:278) skips the
   persona table override for the 12-action config (COM-B handles personas) —
   verify this is the intended behavior per gap map and document/wire
   accordingly.
5. **T4.2 vacuous skip** (`run_stress_tests.py`): `check_t4_2_persona_collapse({})`
   always skips; make it a real check (multi-persona identity-transition ANOVA)
   or a documented limitation.

### 5.3 Validation
- Run all 4 tiers (`run_all.py` or the four check scripts) on the full bootstrap
  table + 4-arm experiment (control/random/fixed/RL).
- C4/C6 are structural FAILs in the 4-state subset only — re-test at full scale
  (full factor variation should resolve them).

---

## 6. Phase 4 — Ship

- Commit table + raw results + logs; PR following repo workflow (linked issue
  without `needs-refinement`, reply-before-resolve, CI green, CodeRabbit
  triage); merge.
- Update `prompt-refinement-log.md` with the full-scale run + constitution
  results.

---

## 7. Verification gates (every code change)

- `uv run ruff format --check`
- `uv run ruff check`
- `uv run ty check --exclude tests/` (4 pre-existing loaders.py diagnostics)
- `uv run pytest` (unit + fast regression)
- Per pilot round: parse-failure count and C3 green.

---

## 8. Risks & fallbacks

| Risk | Mitigation |
|---|---|
| 14,040-call spend wasted if convergence poor | Phase-1 n=6 gate; surface to user |
| Overshoot structural at full scale | Pipeline caps + robust aggregation (r13 decision) |
| Full-scale runtime (~2-3 h) | 50 workers; chunked raw saves |
| T2.2 band ambiguity (210-296 vs 150-450) | Verify against persisted gap map + reference before editing |
| C4/C6 FAILs at full scale | Document; structural factors may resolve them |
