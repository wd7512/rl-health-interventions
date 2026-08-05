# Option B Completion — Full-Scale PEARL Bootstrap Table + Constitution

**Date:** 2026-08-01
**Branch:** `feat/option-b-pipeline-hardening` (6 commits ahead of main `94a5f7a`)
**Status:** Executing
**Related:** #287 (pipeline hardening, closed by #286), future PR for option-B branch

---

## 1. Executive Summary

The Option-B pipeline hardening (literature review, prompt rounds 11–13, frozen
`protocol_fewshot` r13, robust analyzer, bounded retry) is committed but unmerged on
`feat/option-b-pipeline-hardening`. This plan: (1) merges that branch via PR, (2) runs
decision-gated pilot experiments (~624 calls) to pick temperature and validate
sample-count convergence, (3) generates the full 108-state × 13-action bootstrap table
(1,404 cells × 10 samples = 14,040 calls) replacing the stale `pearl_random.json`,
(4) applies the pending constitution Layer-2 fixes and validates all 4 tiers on the full
table, (5) ships.

**Locked decisions (user-confirmed):**

1. **PR first, then full-scale.** Merge the 6-commit option-B branch into main before
   generating the full table; full-scale work proceeds from main on a fresh branch.
2. **Full-scale density: 10 samples/cell** (14,040 calls), gated on a convergence pilot.
3. **Temperature decided by pilot** — temp sweep (0.3 vs 0.7 at n=3) then convergence
   pilot at n=6 on the winning temp.
4. **Pilot sizing: n=3 temp sweep + n=6 convergence** (~624 calls total).
5. **Full table named `tables/pearl_12action/pearl_bootstrap.json`; stale
   `pearl_random.json` deleted** (its burden vocabulary `low/medium/high` is
   incompatible with the config's `none/minor/major`; the table loader globs all
   `*.json` in `table_dir`).

**Context from r13 freeze:** `protocol_fewshot` r13 is FROZEN (mean +223.2 in band,
min −30.0, 47/48 positive, C3 green, 0 parse failures). Remaining high-side overshoot
(max cell +823.8) persists across three ceiling-wording variants — handled at the
pipeline level (per-cell caps + robust aggregation), not by more prompt rounds.

---

## 2. Phase 0 — PR & merge option-B branch

### Scope
- Open PR for `feat/option-b-pipeline-hardening` (6 commits: `5750a2e` literature,
  `84a2413` prompt r11, `864bdb1` robust analyzer, `42b5baa` retry, `2c63b4b` prompt
  r12–13 freeze, `b9fc463` logs + archives).
- Repo workflow: linked issue without `needs-refinement`, reply-before-resolve on all
  review threads (thread-gate), CodeRabbit triage, CI green (lint, test-unit,
  test-integration, build, check-linked-issue, thread-gate).
- Merge into main; create fresh working branch for the remaining work.

### Gates
- `uv run ruff format --check`, `uv run ruff check`, `uv run pytest tests/unit -q`
  (687 tests), `uv run ty check` (4 pre-existing loaders.py diagnostics tolerated).

---

## 3. Phase 1 — Pilot experiments (decision-gated, ~624 calls)

### 3.1 Code change
- Add `--temperature` flag to `scripts/pearl_recalibration/generate_pearl_mini.py`,
  passed through to `batch_complete(temperature=...)` (default stays 0.7).
- Unit test for the flag plumbing.

### 3.2 Temperature sweep (n=3, 156 calls per temp)
- Run frozen r13 prompt at temp 0.3 and 0.7 on the 4-state mini set.
- Compare: mean/median/trimmed lift, min/max cell lift, per-cell variance, positives
  count, parse failures, C3.
- Pick winner: temp with best ceiling behavior (lowest max-cell overshoot) while
  keeping mean in the +150–450 band and ≥47/48 positive.

### 3.3 Convergence pilot (n=6, 312 calls)
- Run winner temp at 6 samples × 52 cells on the same 4 states.
- Question: does the r13 max-cell overshoot (+823.8) persist at n=6 (real signal) or
  shrink (n=3 noise)? Do per-cell means stabilize vs n=3?
- Also gives the per-cell variance evidence supporting the 10-sample full run.

### 3.4 Decision + logging
- Record rounds 14 (temp sweep) and 15 (convergence) in
  `docs/research/prompt-refinement-log.{json,md}`.
- If convergence is poor or overshoot is structural, surface to user before the
  14,040-call spend (fallback: fewer samples, or pipeline cap at analysis).

---

## 4. Phase 2 — Full-scale generation (14,040 calls)

### 4.1 Generator
- New `scripts/pearl_recalibration/generate_pearl_full.py` (mini script stays for
  pilots): `generate_prompts(persona="base", samples_per_cell=10, state_subset=None,
  prompt_variant="protocol_fewshot")` → 108 × 13 = 1,404 cells × 10 = 14,040 calls.
- Same aggregation + bounded retry + raw jsonl save as mini.
- Pipeline-level ceiling handling: per-cell caps in analysis (documented), robust
  aggregation (median/trimmed already in analyzer).
- ~3 h at 50 workers; cost small (deepseek-v4-flash ~$0.2/M tokens).

### 4.2 Output
- Write `tables/pearl_12action/pearl_bootstrap.json`; **delete stale
  `tables/pearl_12action/pearl_random.json`**.
- Validate: `TableValidator` via config load, full-state coverage check
  (all 108 states × 13 actions present, 3 stochastic factors each, probs sum to 1),
  burden vocabulary matches config (`none/minor/major`).
- Config paths already resolve: `config/pearl_constitution_12action.yaml` →
  `../tables/pearl_12action`; `docs/.../pearl_bootstrap.yaml` →
  `../../../../tables/pearl_12action` (config/loader.py:16 resolves relative to
  config file).

---

## 5. Phase 3 — Constitution Layer-2 fixes + validation

### 5.1 Persist gap map
- `docs/research/constitution-gaps.md` — currently the gap map lives only in
  conversation; persist first so fixes are grounded.

### 5.2 Fixes (verify each against the persisted gap map)
1. **T2.3 ordering** (`run_distribution_check.py:86`): change strict
   `RL ≥ Fixed ≥ Random > Control` to `RL > (Fixed ≈ Random ≈ Control)` per gap map.
2. **T2.2 band** (`run_distribution_check.py:58`): gap map says 210–296 band;
   current check uses 150–450. **Verify intended band before changing** — resolve
   against PEARL reference (Δ effect target) and r13 ladder results.
3. **μ 5,618 vs 5,580**: resolve discrepancy between `pearl_reference.json`
   (`mean_baseline_steps: 5618.2`) and the 5,580 figure referenced in
   `utils.py` `_RECENT_STEPS_MEAN_MIDPOINT` comment; make check constants derive from
   `load_reference()`.
4. **persona_name wiring**: `load_constitution_config` (utils.py:278) skips persona
   table override for 12-action config (COM-B handles personas) — verify this is the
   intended behavior per gap map and document/wire accordingly.
5. **T4.2 vacuous skip**: `run_stress_tests.py` `check_t4_2_persona_collapse({})`
   always skips; make it a real check (multi-persona identity-transition ANOVA) or
   documented limitation.

### 5.3 Validation
- Run all 4 tiers (`run_all.py`, or the four check scripts) on the full bootstrap
  table + 4-arm experiment (control/random/fixed/RL).
- Record results; C4/C6 known FAILs in the 4-state subset should be re-tested at full
  scale (full factor variation may resolve them).

---

## 6. Phase 4 — Ship

- Commit table + raw results + logs; PR following repo workflow; merge.
- Update `prompt-refinement-log.md` with full-scale run + constitution results.

---

## 7. Risks & fallbacks

| Risk | Mitigation |
|---|---|
| 14,040-call spend wasted if convergence poor | Phase-1 n=6 gate; surface to user |
| Overshoot structural at full scale | Pipeline caps + robust aggregation (r13 decision) |
| Full-scale runtime (~3 h) | 50 workers; chunked raw saves; resumable |
| T2.2 band ambiguity (210–296 vs 150–450) | Verify against persisted gap map + reference before editing |
| C4/C6 FAILs at full scale | Document; structural factors may resolve them |
