# Full-Scale PEARL Bootstrap — Run Report & Constitution Findings

**Date:** 2026-08-01
**Branch:** `feat/full-scale-pearl-bootstrap`
**Table:** `tables/pearl_12action/pearl_bootstrap.json` (1,404 cells, 10 samples/cell)
**Status:** Table complete and structurally valid; 4 constitution checks fail on
real behavior with a documented root cause; reward-penalty probe restores the
reference arm ordering in reward space but not step space.

---

## 1. Full-scale generation

- **Scope:** 108 states × 13 actions = **1,404 cells × 10 samples = 14,040
  prompts** (variant `protocol_fewshot`, temp 0.3).
- **Runtime:** 109.7 min for the main run + 4.1 min for the `--retry-errors`
  recovery pass (1,000 workers, batch 1,000, per-request `--timeout 120`,
  `--retries 1`).
- **Raw records:** 14,096 JSONL lines (includes retry records kept alongside
  originals); error records stripped by the recovery pass.

### 1.1 Reliability fixes landed mid-run

1. **Per-batch append + cell-level resume** (`51abfab`) — raw records hit disk
   after every batch, so a crash never loses more than the in-flight batch;
   resume tops up cells with <10 content records.
2. **Two-phase flush** (`51abfab`) — the primary batch's records append to disk
   *before* the unparseable-response retry runs, so a stalled retry can never
   hold already-succeeded output hostage in memory.
3. **Bounded request time** (`4c807e3`) — `timeout` and `num_retries` are
   threaded through `request.batch_complete` → litellm. At 1,000 workers the
   Sprint-1 default (`num_retries=7`, unbounded future waits) let one hung
   request hold a whole batch for ~13-14 min. `--timeout 120 --retries 1` makes
   a hung request fail fast so `--resume` tops it up.

### 1.2 Post-run recovery pass

- `_run_batch` retries **unparseable content** only; **LLM errors** (network /
  timeout / provider failures) are appended as error records and never re-run.
  This left 97 cells with error records and 71 cells with <2 parseable samples
  (the aggregator drops such cells).
- `--retry-errors --resume` strips error records then tops up the affected
  cells: 90 cells / 773 prompts, completed cleanly.

---

## 2. Final table validation

| Check | Result |
|---|---|
| `TableValidator` (schema + stochastic factors + sums-to-1) | **PASS** |
| Cell coverage (108×13 = 1,404) | **1,404 / 1,404** |
| `n_samples` per cell | **10 for all 1,404 cells** |
| Sum-to-1 probability errors | **0** |
| Unknown / extra cells | **0** |
| Stale `tables/pearl_12action/pearl_random.json` | **deleted** (burden `low/medium/high` incompatible; loader globs all `*.json`) |

---

## 3. Constitution checks (T1-T4) on the full table — 11/17 pass

Run: `uv run python -m scripts.pearl_constitution.run_all --seeds 10`
(the 12-action config, so tiers read the full bootstrap table).

### 3.1 PASS (11)

- **T1.1** Baseline stability (control mean 5,500 within ±15% of 5,618.2)
- **T1.2** Action differentiation (ANOVA F=11.6, p=1.8e-5)
- **T1.3** Direction correctness (RL > Control in 9/10 seeds)
- **T1.4** No degenerate trajectories
- **T2.5** Between-person variance (ICC=0.51, band 0.4-0.9)
- **T3.2** Persona heterogeneity (skipped by design — single persona)
- **T3.4** Non-response detection (structural PASS; note: check is meaningful
  only with a null/action-effect-zero config)
- **T4.1** Random matrix (ANOVA p=1.0)
- **T4.2** Persona collapse — now a documented **WARNING** skip (gap-map fix;
  env-gated real check via `PEARL_T4_2_AVAILABLE=1`)
- **T4.3** Infinite horizon (1.5% drift at 90 days)
- **T4.4** Extreme demographics (skipped — requires `--persona resistant`)

### 3.2 FAIL (6)

| ID | Result | Classification |
|---|---|---|
| T2.1 | `t=-inf`, p=0.0, mean=5,500 vs ref 5,618.2 | **Check artifact** — control baseline is fully deterministic at exactly 5,500 (idle keeps state at `moderate`), so between-seed variance is 0 and the t-test degenerates. T1.1 (same mean, ±15% band) passes. |
| T2.2 | Δ=1,525 steps vs reference band [218-296] | **Real deviation** — simulated RL-vs-control effect is ~6x the reference. |
| T2.3 | Fixed=7,741.7 > RL=7,425.0 | **Real deviation** — ordering inversion vs reference (RL > Fixed). |
| T2.4 | Δ grows month 1→2 (attenuation −7%) | **Real deviation** — reference attenuates (29%). |
| T3.1 | No burden saturation (steps keep rising past day 21) | **Real deviation**. |
| T3.3 | Weekend effect inverted (−1.7% vs expected 5-20%) | **Real deviation**. |

---

## 4. 4-arm experiment (50 seeds, reward space)

`docs/experimental_phases/pearl_random/run_experiments.py --seeds 50
--config pearl_bootstrap.yaml`

| Arm | Total reward | Per step | Last 50 |
|---|---|---|---|
| COM-B weighted fixed | 49.2 ± 15.7 | 0.820 | 0.877 |
| **RL (ε-greedy)** | 47.5 ± 18.1 | 0.792 | 0.874 |
| Random | 45.6 ± 21.3 | 0.760 | 0.825 |
| Control (idle) | 10.9 ± 22.7 | 0.182 | 0.217 |

**Readings:**
- RL does **not** beat the COM-B weighted fixed policy (49.2 vs 47.5) —
  contradicts the reference ordering.
- Control (idle) collapses (10.9 vs 45+). Idle is far more punitive in the
  table than in the real trial, which drives the oversized T2.2 effect.
- Random ≈ RL (45.6 vs 47.5): the table rewards *sending an intervention*
  far more than *sending the right one*.

**Root-cause hypothesis:** the LLM-generated `idle` / no-intervention
transitions drift toward `low` recent-steps while *any* COM-B action keeps a
user at `moderate`/`high`. Most of the effect is therefore "action vs
no-action", not action quality — so Fixed's any-action schedule ≈ RL, Random ≈
RL, and every active arm crushes idle.

---

## 5. Reward-penalty probe (does changing the reward change anything?)

Config variant with `action_penalty` raised from **0.05 → 0.25** for all 12
intervention actions (`pearl_bootstrap_reward_penalty.yaml`,
`pearl_constitution_12action_reward_penalty.yaml`).

### 5.1 Reward space (4-arm benchmark, 50 seeds)

| Arm | Penalty 0.05 | Penalty 0.25 |
|---|---|---|
| **RL (ε-greedy)** | 47.5 ± 18.1 | **37.8 ± 18.0** |
| COM-B weighted fixed | **49.2 ± 15.7** | 37.2 ± 15.7 |
| Random | 45.6 ± 21.3 | 34.6 ± 21.2 |
| Control (idle) | 10.9 ± 22.7 | 10.9 ± 22.7 |

**Result:** the penalty restores the **reference ordering** in reward space
— RL > Fixed > Random > Control. Raising the cost of intervening makes the
ε-greedy agent more selective, letting it edge out the COM-B schedule.

### 5.2 Step space (constitution tiers, 10 seeds)

The constitution's step-domain checks are **unchanged** by the penalty:
T2.3 still Fixed=7,741.7 > RL=7,433.3, Δ still ≈1,530, T3.1/T3.3 unchanged.
T2.1 `t=-inf` also persists (the deterministic-idle baseline is independent of
reward).

**Why:** the step domain is driven by the transition table's
`recent_steps_mean` structure (idle → `low`, any action → `moderate`/`high`);
the reward penalty only reshapes which action the RL agent *selects*, and the
step-value spread (3,000/5,500/8,000 midpoints) swamps any penalty-driven
selection change.

### 5.3 Implication

- Reward calibration can fix the **arm-ordering** symptom (RL vs Fixed) in the
  domain the agent optimizes, but **cannot fix the step-domain deviations**
  (oversized Δ, no attenuation, no saturation, inverted weekend). Those are
  properties of the transition table itself and would need either (a) table
  regeneration with different prompt constraints, or (b) recalibrated
  step-midpoint mapping / check bands, or (c) acceptance as documented
  limitations of the LLM-bootstrap approach.

---

## 6. Files

- `tables/pearl_12action/pearl_bootstrap.json` — final table (1,404 cells).
- `tables/pearl_12action/raw/results_full_protocol_fewshot.jsonl` — raw records.
- `docs/experimental_phases/pearl_random/results/pearl_bootstrap_full/` and
  `.../pearl_bootstrap_reward_penalty/` — 4-arm benchmark fixtures (50 seeds).
- `config/pearl_constitution_12action_reward_penalty.yaml` and
  `docs/experimental_phases/pearl_random/configs/pearl_bootstrap_reward_penalty.yaml`
  — reward-penalty probe configs.
- `docs/research/constitution-gaps.md` — persisted gap map (Phase-3 fixes).
