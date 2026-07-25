# PR #266 Review — Atomic PR Decomposition & Roadmap

**Related:** [#252](https://github.com/wd7512/rl-health-interventions/issues/252), [#267](https://github.com/wd7512/rl-health-interventions/issues/267)
**PR:** [#266](https://github.com/wd7512/rl-health-interventions/pull/266)
**Date:** 2026-07-25
**Status:** Planning

---

## 1. Executive Summary

PR #266 (21 commits, ~5,500 lines, single day) implements the PEARL-matched config
from Issue #252. A full commit-by-commit review found 10 bugs (3 critical), architecture
reversals, and mixed concerns. This document decomposes the work into 7 atomic PRs with
clear dependency chains, identifies remaining gaps, and provides an adversarial
retrospective.

**Key finding:** The epsilon sweep (commit 16) conclusively showed random transitions
produce a ~54% sustained burden floor regardless of agent policy. The RL agent cannot
learn to manage burden because burden is determined by action-independent stochastic
transitions. Future work must focus on structured transition models.

---

## 2. Commit-by-Commit Summary

| # | SHA | Description | Scope | Key Issue |
|---|-----|-------------|-------|-----------|
| 1 | `52cef8b` | Phase 2 plan document (683 lines) | docs | §15.3 superseded during implementation |
| 2 | `d33225c` | COM-B agent + scaffold (735 lines) | core+config+tests | Copilot-authored, lint issues |
| 3 | `582a90e` | Polish lint/formatting (39 lines) | lint | 3-min follow-up to #2 |
| 4 | `260bdb1` | RandomTransitionSA + Bayesian burden (3,042 lines) | MAJOR | Class removed 27 min later |
| 5 | `dedc712` | Consolidate + rename bootstrap→table_transition (48 files) | refactor | Breaking rename across 35+ configs |
| 6 | `b91b967` | CI standardization (8 lines) | CI | Safe, mechanical |
| 7 | `bd1d5b2` | Per-factor table format + path fixes (1,020 lines) | BUG FIX | Flat format only sampled one factor |
| 8 | `4c6dff7` | Regenerate fixture 5→50 seeds (1 file) | test data | Still had Windows path bug |
| 9 | `8cea0e0` | Key ordering + historical context fix (137 lines) | BUG FIX | Burden mechanism was disabled |
| 10 | `c81870c` | CodeRabbit review comments (134 lines) | multi-fix | Day-boundary state propagation |
| 11 | `fa574ce` | Rename Bayesian P-success → P-success (27 lines) | docs | Cosmetic |
| 12 | `26fec22` | Visualizations + _reject_fields fix (381 lines) | plots+schema | Bool default bug |
| 13 | `698aec8` | Trajectory export + plot rewrite (84,640 lines) | infrastructure | 84k-line JSON committed |
| 14 | `8c30d10` | Fix unused import (2 lines) | cleanup | Minor |
| 15 | `f7b1a10` | Per-step posterior burden formula (5,100 lines) | CRITICAL | Private API coupling |
| 16 | `b33272c` | Epsilon sweep study (299 lines) | experiment | Confirms RL can't learn |
| 17 | `73b2087` | Contextual benchmark (281 lines) | experiment | Context useless on random |
| 18 | `d68547b` | Type error fixes (10 lines) | BUG FIX | Swapped destructuring |
| 19 | `65307aa` | Agg order + numpy import fix (32 lines) | BUG FIX | Edit-conflict bugs |
| 20 | `d340935` | Explicit mechanism field (44 lines) | config | Should have been day-1 |
| 21 | `3ace7fc` | Test suite split (113s→8s) | tooling | DX improvement |

---

## 3. Bug Inventory

### Critical (shipped, later fixed in same PR)

| # | Bug | Introduced | Fixed | Impact |
|---|-----|------------|-------|--------|
| 1 | `sorted()` vs config order for state keys — all P-success lookups returned 0.5 | commit 4 | commit 9 | Burden mechanism disabled |
| 2 | Swapped destructuring in `run_agent()` — returns trajectories instead of rewards | commit 13 | commit 18 | Silent data corruption |
| 3 | Day-boundary state propagation missing — within-day uses pre-boundary state | commit 4 | commit 10 | Transitions ignore day boundaries |

### High

| # | Bug | Introduced | Fixed | Impact |
|---|-----|------------|-------|--------|
| 4 | Flat format sampled one factor only | commit 4 | commit 7 | Other stochastic factors never updated |
| 5 | Historical state context lost — used current state for past actions | commit 4 | commit 9 | Wrong P-success lookups |
| 6 | `_reject_fields` bool defaults — `False is not None` always True | commit 2 | commit 12 | False positives for bool fields |

### Medium

| # | Bug | Introduced | Fixed | Impact |
|---|-----|------------|-------|--------|
| 7 | table_dir paths resolved from wrong base | commit 4 | commit 7 | Configs fail to load tables |
| 8 | `_REPO_ROOT` 3 levels instead of 4 | commit 4 | commit 7 | generate_tables.py fails |
| 9 | `matplotlib.use("Agg")` after pyplot import | commit 12 | commit 19 | Non-headless backend loaded |
| 10 | numpy import lost during edit | commit 13 | commit 19 | NameError crash |

---

## 4. Atomic PR Decomposition

### PR-A: Com-B Weighted Fixed Agent + Config
**Commits:** 2, 3 | **Risk:** Low | **Deps:** None

**Files:**
- `src/rl_health_interventions/agents/fixed.py` — `ComBWeightedFixedAgent`
- `config/pearl/comb_scores.json` — 5 personas
- `src/rl_health_interventions/config/schemas.py` — agent field validation
- `tests/unit/agents/test_fixed_agent.py` — agent tests
- `tests/unit/agents/test_agents_registry.py` — registry tests

**Deliverables:** Agent class, schema, COM-B scores, unit tests.

### PR-B: Table Transition Rename + Per-Factor Format
**Commits:** 4 (partial), 5, 7 | **Risk:** Medium (breaking) | **Deps:** None

**Files:**
- `src/rl_health_interventions/transitions/table_transition.py` (renamed from bootstrap)
- `src/rl_health_interventions/config/schemas.py` — `_KNOWN_TRANSITION_TYPES`
- 35+ YAML configs — `type: bootstrap` → `type: table_transition`
- `docs/experimental_phases/pearl_random/generate_tables.py`
- `tables/pearl_12action/` — generated transition tables
- `tests/unit/transitions/test_table_transition.py`

**Deliverables:** Renamed transition model, per-factor format support, generation script, tables.

### PR-C: PEARL Config Files + Experiment Runner
**Commits:** 2 (partial), 5 (partial) | **Risk:** Low | **Deps:** PR-A, PR-B

**Files:**
- `docs/experimental_phases/pearl_random/configs/pearl_random.yaml`
- `docs/experimental_phases/pearl_random/configs/pearl_bootstrap.yaml`
- `docs/experimental_phases/pearl_random/_shared.py`
- `docs/experimental_phases/pearl_random/run_experiments.py`
- `tests/regression/test_pearl_random.py`
- `docs/experimental_phases/pearl_random/results/pearl_random.json`

**Deliverables:** Config files, runner, regression test.

### PR-D: Posterior Burden Mechanism
**Commits:** 4 (partial), 9, 10 (partial), 15, 20 | **Risk:** HIGH | **Deps:** PR-B

**Files:**
- `src/rl_health_interventions/environment.py` — `_compute_burden_chance()`
- `src/rl_health_interventions/config/schemas.py` — `mechanism` field
- `tests/unit/test_bayesian_burden.py`

**Deliverables:** Formula 3 implementation, mechanism config field, 3-tuple action history, unit tests.

**Note:** This is the core algorithmic contribution. 5+ bugs were found and fixed in this
area during PR #266. Recommend extra review scrutiny.

### PR-E: Documentation + D15 + Constitution Corrections
**Commits:** 4 (partial), 10 (partial), 11 | **Risk:** Low | **Deps:** None (parallelizable)

**Files:**
- `docs/research/decision-catalogue.md` — D15 entry
- `docs/research/pearl-constitution.md` — arm mappings, baseline, T2.3
- `docs/experimental_phases/pearl_random/README.md`
- `docs/plans/phase-2-pearl-matched-config.md` — status updates

**Deliverables:** Feature selection ADR, constitution corrections, experiment README.

### PR-F: Experiment Infrastructure + Visualizations
**Commits:** 12, 13, 16, 17 | **Risk:** Low | **Deps:** PR-C, PR-D

**Files:**
- `docs/experimental_phases/pearl_random/plots.py`
- `docs/experimental_phases/pearl_random/sweep_epsilon.py`
- `docs/experimental_phases/pearl_random/bench_contextual.py`
- `docs/experimental_phases/pearl_random/images/` — generated plots

**Deliverables:** Trajectory export, 5 visualizations, epsilon sweep, contextual benchmark.

**Note:** Experimental scripts. The epsilon sweep results (RL can't learn on random
transitions) are a critical finding that should inform future direction.

### PR-G: Dev Tooling + CI
**Commits:** 6, 14, 18, 19, 21 | **Risk:** Low | **Deps:** None (parallelizable)

**Files:**
- `.github/workflows/ci.yml` — `uv sync --all-extras`
- `regression-suite/` — extracted slow tests
- `docs/experimental_phases/pearl_random/_shared.py` — swapped destructuring fix
- `docs/experimental_phases/pearl_random/plots.py` — Agg order, numpy import
- `docs/experimental_phases/pearl_random/generate_tables.py` — unused import

**Deliverables:** CI standardization, test split (113s→8s), bug fixes.

---

## 5. Dependency Graph

```
PR-G (Tooling)  ──────────────────────────────────────────┐
PR-E (Docs)     ──────────────────────────────────────────┤
PR-A (Agent)    ───┐                                      │
                   ├── PR-C (Configs) ──┐                 │
PR-B (Transition) ─┤                    ├── PR-F (Experiments)
                   ├── PR-D (Burden) ──┘                 │
                   │                                      │
                   └──────────────────────────────────────┘
```

**Critical path:** PR-B → PR-D → PR-F
**Parallelizable:** PR-G, PR-E, PR-A (no deps on each other)

---

## 6. Remaining Gaps

### From Plan (§15)

| Item | Status | Notes |
|------|--------|-------|
| Bootstrap config variant | ⚠️ Identical to pearl_random | Should differ (e.g., different table source) |
| Attrition model | ❌ Not implemented | PEARL had 42.7% withdrawal |
| Relative-change reward formula | ❌ Deferred | `step_reward - action_penalty` is placeholder |
| Baseline period | ❌ Not modeled | PEARL used 30-day pre-study window |

### From PR #252

| Item | Status | Notes |
|------|--------|-------|
| Config file | ✅ Created | In experimental_phases/, not config/ |
| State variables | ✅ 5 dynamic vars | Simplified from PEARL's 20+ |
| Action space | ✅ 12 actions | Matches PEARL design |
| 4-arm structure | ✅ Implemented | All 4 arms working |
| Episode length | ✅ 60 days | Matches PEARL |
| Baseline period | ❌ Not modeled | Gap |
| Config loads | ✅ Confirmed | Schema validation passes |
| All arms valid | ✅ Confirmed | No crashes, no NaN |
| Schema validation | ✅ Passes | |
| Design decisions | ✅ Documented | Extensive plan doc |

### Structural Gap: pearl_constitution.yaml

The validation scripts (`scripts/pearl_constitution/`) load `config/pearl_constitution.yaml`,
which still uses:
- 5 steps/day (should be 1)
- 4 actions (should be 12)
- Thompson Sampling (should be epsilon-greedy)
- Old state variables (step_bin, sleep, not PEARL features)

This means the 4-tier constitution validation is running against the wrong simulation
design. **This is the highest-priority fix after the atomic PR decomposition.**

---

## 7. Adversarial Retrospective

### What I Would Do Differently

1. **Build burden mechanism first, config files second.** The initial implementation
   shipped config + runner + tests with naive action counting, producing degenerate
   results that needed 3 fixture regenerations.

2. **Never create RandomTransitionSA.** Design the transition infrastructure first:
   add per-factor format to TableTransition, create generation script with pure numpy,
   generate tables. The class existed for 27 minutes — architecture wasn't designed.

3. **Fix bugs in isolation.** The key ordering bug (commit 9) disabled the entire
   burden mechanism but was only found after the per-factor format fix (commit 7)
   exposed it. Each structural change should be verified end-to-end before proceeding.

4. **Separate experimental scripts from core PRs.** Epsilon sweep and contextual
   benchmark (commits 16-17) are research exploration that mixed with core bug fixes.

5. **Don't commit 84k-line trajectory files.** Use `.gitignore` and CI artifacts.

6. **Add `mechanism` field from day 1.** The implicit `_p_success` gate required
   3 refactors. Explicit config fields are always better than inferred behavior.

### What Was Done Well

1. **The plan document (commit 1) was excellent.** 683 lines of grounded design with
   14 decisions, gap analysis, and risks. It evolved to capture implementation gaps.

2. **Test split (commit 21) dramatically improved DX.** 113s→8s is 14x faster.

3. **Posterior burden formula (commit 15) is the right algorithm.** Per-step Bayesian
   posterior matches PEARL's actual Formula 3.

4. **Epsilon sweep produced valuable negative results.** The ~54% burden floor finding
   is critical for interpreting future results and steering toward structured transitions.

---

## 8. Recommended Execution Order

```
Phase 1 (parallel):  PR-G + PR-E + PR-A
Phase 2 (sequential): PR-B
Phase 3 (parallel):  PR-C + PR-D
Phase 4:             PR-F
Phase 5:             Fix pearl_constitution.yaml to use PEARL config
Phase 6:             Structured transition models (bootstrap/learned)
```

**After Phase 4:** Run the full PEARL Constitution validation (4 tiers, 16 checks)
against the PEARL-matched config. This is the milestone where we know whether the
simulator can reproduce RCT distributions.

---

## 9. References

- Lee et al. (2025). "A Personalized Exercise Assistant using Reinforcement Learning (PEARL)." arXiv:2508.10060.
- Phase 1 deep analysis: `docs/research/recreations/pearl-rct-2025/pearl-deep-analysis.md`
- Phase 2 plan: `docs/plans/phase-2-pearl-matched-config.md`
- Constitution spec: `docs/research/pearl-constitution.md`
- Decision catalogue: `docs/research/decision-catalogue.md`
