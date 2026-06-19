# Epic: State-Action-Reward Extension

**Epic branch:** `epic/state-action-reward`

**Parent:** `main`

**Status:** `[ ]` In progress

---

## Vision

Extend the MVP into a configurable simulation with richer actions,
continuous state dimensions, and delayed reward signals. Deliver 7
experiment configurations covering every combination.

---

## Goals

- [ ] **G1 — Rich actions:** Actions carry metadata (cost, burden) usable by reward/transition
- [ ] **G2 — Continuous state:** State embeds continuous patient features (steps, weight, time) beyond activity label
- [ ] **G3 — Delayed reward:** Reward supports multi-timescale signals (interval bonuses) alongside per-step reward
- [ ] **G4 — 7 experiment configs:** Every pairwise and full combination of G1/G2/G3 runs end-to-end

---

## Gates

| Gate | Check | When |
|------|-------|------|
| G-BK | `config/rule_based.yaml` produces identical results | Every PR |
| G-EXP | At least one experiment config runs end-to-end per goal | Phase 1 close |
| G-INT | All pairwise combo configs load + run without error | Phase 2 close |
| G-FULL | Full combo config + cross-experiment script works | Phase 3 close |

---

## Branch Structure

```
main ── epic/state-action-reward
          ├─ pr1a ─┐
          ├─ pr1b ─┼─ pr4(1a1b) pr5(1a1c) pr6(1b1c) ── pr7(full)
          └─ pr1c ─┘
```

## Config Targets

| # | Config | Goals |
|---|--------|-------|
| 1 | `config/experiments/1a_actions_only.yaml` | G1 |
| 2 | `config/experiments/1b_states_only.yaml` | G2 |
| 3 | `config/experiments/1c_reward_only.yaml` | G3 |
| 4 | `config/experiments/1a_1b.yaml` | G1+G2 |
| 5 | `config/experiments/1a_1c.yaml` | G1+G3 |
| 6 | `config/experiments/1b_1c.yaml` | G2+G3 |
| 7 | `config/experiments/full.yaml` | G1+G2+G3 |

---

## Phase 1 — Individual Extensions (3 PRs, parallel from main)

Each branches from `main`, extends one dimension, PRs into `epic`.

### PR 1a: `feat/1a-actions` — G1 Rich actions

**Acceptance:**
- [ ] Actions config supports structured entries (name + metadata)
- [ ] Duplicate action names rejected
- [ ] MVP string-list configs still parse identically
- [ ] Transition/reward models can query metadata by action name
- [ ] Config `1a_actions_only.yaml` runs end-to-end

**Modules:** `schemas.py`, agents, transition config

---

### PR 1b: `feat/1b-states` — G2 Continuous state

**Acceptance:**
- [ ] `StateView` carries optional continuous fields defaulting to `None` (MVP backward compat)
- [ ] Transition model ABC accepts `StateView` instead of bare string
- [ ] Reward handler ABC accepts `StateView`
- [ ] MVP-config environments produce identical results
- [ ] Continuous state fields evolve per-step when configured
- [ ] Deterministic fields (time_of_day, day_of_week) derived from step counters
- [ ] Config `1b_states_only.yaml` runs end-to-end

**Modules:** `state.py`, `transitions/_base.py`, `transitions/rule_based.py`, `rewards/_base.py`, `rewards/compound.py`, `environment.py`

---

### PR 1c: `feat/1c-reward` — G3 Delayed reward

**Acceptance:**
- [ ] Config can opt into multi-timescale reward (default remains simple/MVP)
- [ ] Simple mode produces identical per-step rewards to current MVP
- [ ] Delayed bonus fires at configurable intervals
- [ ] Bonus can be flat or activity-scaled (implementation decides)
- [ ] Config `1c_reward_only.yaml` runs end-to-end

**Modules:** `schemas.py`, `rewards/compound.py`, `environment.py`

---

## Phase 2 — Pairwise Integrations (3 PRs, from epic after Phase 1)

### PR 4: `int/1a-1b` — G1+G2

- [ ] Config `1a_1b.yaml` runs end-to-end with rich actions + continuous state
- [ ] State dynamics respond to action metadata

### PR 5: `int/1a-1c` — G1+G3

- [ ] Config `1a_1c.yaml` runs end-to-end with rich actions + delayed reward
- [ ] Reward computation consumes action metadata (e.g., action cost)

### PR 6: `int/1b-1c` — G2+G3

- [ ] Config `1b_1c.yaml` runs end-to-end with continuous state + delayed reward
- [ ] Delayed reward can reference continuous state features

---

## Phase 3 — Full Integration

### PR 7: `int/full` — G1+G2+G3

- [ ] Config `full.yaml` runs end-to-end with all three extensions
- [ ] `scripts/run_cross_experiment.py` runs all 7 configs, outputs comparison
- [ ] Integration test: all 7 configs load + run without error

---

## Verification Per PR

```bash
uv run ruff format --check .
uv run ruff check
uv run ty check --exclude tests/
uv run pytest
```

---

## Risks

| Risk | Mitigation |
|------|-----------|
| ABC signature change (PR 1b) touches ~15 test files | Mechanical wrapping; each file change is 1-2 lines |
| Regression fixture churn from RNG seed interaction | Freeze MVP config test path; assert MVP config independently |
| Parallel PRs to `schemas.py` may conflict | PR 1b branches after 1a merges or rebases onto epic |
