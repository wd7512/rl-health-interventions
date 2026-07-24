# Phase 2 — PEARL-Matched Config (#252)

**Branch:** `feature/252-pearl-matched-config`
**Date:** 2026-07-24
**Status:** Partial implementation — fix plan below
**Source:** Conversation anchored summary + Phase 1 deep analysis + Figure 12 feature importance

---

## 1. Overview

Phase 2 aligns the simulation config and code to match the PEARL RCT design (Lee 2025) as
determined by the Phase 1 deep paper analysis. The current config (`pearl_constitution.yaml`)
has 5+ structural mismatches against the actual PEARL study.

### Gap Summary

| Dimension | Current Simulator | PEARL (Paper) |
|-----------|-------------------|---------------|
| Steps per day | 5 | 1 |
| Actions | 4 (idle, movement_suggestion, goal_reminder, journal) | 12 (6 COM-B themes × 2 delivery times) |
| RL algorithm | Thompson Sampling | ε-greedy C-MAB (ε=0.7-0.8) |
| Reward function | `α * step_bin + (1-α) * sleep - penalty` | `(steps_24h - B(M,W)) / B(M,W)` |
| Baseline period | 7 days | 30-day pre-study window |
| State variables | 4 categorical (step_bin, sleep, day_of_week, burden) | 20+ features (Table 7) |
| Fixed arm | `FixedAgent(action="movement_suggestion")` | COM-B barrier-score weighted multinomial |
| Burden model | Count of non-idle actions in last 3 steps | P-success (7-day failure window) |

### Scope — Implementation Status

| Item | Status | Notes |
|------|--------|-------|
| `ComBWeightedFixedAgent` | ✅ Done | `agents/fixed.py`, registered, 3 unit tests |
| `config/pearl/comb_scores.json` | ✅ Done | 5 personas, matches plan §9 exactly |
| `pearl_random.yaml` config | ✅ Done | 12 actions, 4 agents, reward formula |
| Experiment runner + shared utils | ✅ Done | `run_experiments.py`, `_shared.py` |
| Schema + registry updates | ✅ Done | `comb_weighted_fixed` in known types |
| Regression tests + golden fixtures | ✅ Done | `test_pearl_random.py`, `pearl_random.json` |
| **P-success burden** | ✅ Done | `environment.py` — precomputed P(s\|s,a) via transition tables; fixes #252 |
| **12-action transition tables** | ✅ Done | `tables/pearl_12action/` — per-factor format with `day_boundary.json` + `within_day_0.json` |
| **D15 decision catalogue entry** | ✅ Done | `docs/research/decision-catalogue.md:302` — feature selection entry |
| **Constitution corrections** | ✅ Done | `docs/research/pearl-constitution.md` — T1.1, T2.1, T2.3, Arm Mapping |
| **README for pearl_random** | ✅ Done | `docs/experimental_phases/pearl_random/README.md` |
| **Bootstrap config variant** | ❌ Not done | Plan §10 Step 4b not implemented |

---

## 2. Key Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Feature selection | weight >= 0.5 boundary, 8 features | Natural break (next is 0.32, 47% drop). See §3. |
| 2 | State space size | 108 states (5 dynamic vars) | 3 × 2 × 3 × 2 × 3 = 108. Tractable for bootstrap. |
| 3 | Static features | Persona context (not in state) | Static attrs don't change; transitions depend only on dynamic vars. |
| 4 | Burden mechanism | P-success from transition tables | Probability action & idle distributions differ via lookup. See §6. |
| 5 | Burden window | 7 days (1 week) | Carries across days; naturally decays as old failures fall out of window. |
| 6 | Burden mapping | 0-2→low, 3-5→medium, 6+→high | Derived from observed PEARL effect sizes (200-300 steps). |
| 7 | RL algorithm | ε-greedy C-MAB | PEARL uses ε-greedy with ε=0.7-0.8. |
| 8 | Epsilon convention | Keep existing code (ε=explore prob) | PEARL ε=0.7 maps to our ε=0.3. Documented in config. |
| 9 | Fixed arm | COM-B barrier-score weighted multinomial | Barrier = 5 - Likert score; sampled from multinomial. |
| 10 | Reward formula | `step_reward - action_penalty` (phase 1) | Simpler for random-transition testing; relative change formula deferred to bootstrap phase. |
| 11 | Baseline | Per-persona constant | `baseline_steps_mean` loaded from synthetic persona file. |
| 12 | COM-B survey | Synthetic JSON file per persona | 6 themes × 1-5 Likert scale. See §9. |
| 13 | Transition tables | Random for testing; bootstrap deferred | Use `transition_model: type: random` (like sprint1_random) until tables are regenerated. |
| 14 | Notification outcome | Tracked internally, not in state | Does not inflate state space. Precomputed from transition tables. |

---

## 3. Feature Selection ADR (D15)

### Source
Figure 12, Lee 2025 PEARL RCT — XGBoost model weights across 7,711 participants, 60 days,
12 actions. Extracted via pymupdf from `lee-2025-pearl-rct.pdf`.

### Boundary: weight >= 0.5
Natural break point — next highest weight is 0.32 (47% drop). Captures ~95% of total
model importance.

### Selected Features (8 total = 5 static + 3 dynamic)

| # | Feature | Weight | Type | Category |
|---|---------|--------|------|----------|
| 1 | Pre-study steps mean | 1.23 | Static | Persona |
| 2 | Weight | 1.19 | Static | Persona |
| 3 | Age | 1.17 | Static | Persona |
| 4 | Pre-study walk pattern: Low vs High | 1.13 | Static | Persona |
| 5 | Pre-study steps StdDev | 1.02 | Static | Persona |
| 6 | Recent steps mean | 0.88 | Dynamic | State |
| 7 | Recent walk pattern: Low vs High | 0.70 | Dynamic | State |
| 8 | Avg steps Morning: High | 0.63 | Dynamic | State |

### Dropped Features (weight < 0.5)

| Feature | Weight | Reason |
|---------|--------|--------|
| gender_FEMALE | 0.32 | Below threshold |
| Pre-study walk pattern: Regular vs Irregular | 0.32 | Below threshold |
| Recent walk pattern: Regular vs Irregular | 0.31 | Below threshold |
| Percentage of missing data | 0.29 | Not applicable in simulation |
| Recent steps StdDev | 0.25 | Below threshold |
| Two days back daily steps | 0.24 | Recent steps mean captures this |
| Slope of regression line | 0.24 | Recent steps mean captures trend |
| dyn_activity_type_Walking | 0.16 | Below threshold |
| height | 0.08 | Negligible |
| Morning (COM-B survey) | 0.06 | Negligible |
| Ability (COM-B survey) | 0.06 | Negligible |

### Static Features as Persona Context

Static features (baseline_steps_mean, baseline_walk_pattern, weight, age, baseline_steps_std)
are persona attributes set at episode start. They do not change during the simulation and are
not part of the agent's observable state. This keeps the state space at 108 and prevents the
full 105,000-state combinatorial explosion.

---

## 4. State Space Design

### Dynamic State (agent observes — 108 states)

| Variable | Values | States | Behavior |
|----------|--------|--------|----------|
| recent_steps_mean | low, moderate, high | 3 | Updates daily from step totals |
| recent_walk_pattern | low, high | 2 | Classification from recent steps |
| morning_steps_ratio | morning, balanced, evening | 3 | Time-of-day step split |
| day_of_week | weekday, weekend | 2 | Cyclic (7-day pattern) |
| burden | low, medium, high | 3 | P-success (7-day window) |

**Total:** 3 × 2 × 3 × 2 × 3 = **108 states**

### Transition Probability Size

| Metric | Value |
|--------|-------|
| States | 108 |
| Actions | 12 + idle = 13 |
| State-action pairs | 1,404 |
| Next state distribution | 108 probs per pair |
| Total transition entries | **151,632** |

Tractable for bootstrap transition tables. Each persona table: ~1,404 rows.

### Persona Constants (not in state)

| Variable | Type | Purpose |
|----------|------|---------|
| baseline_steps_mean | float | Reward baseline, per-persona |
| baseline_walk_pattern | str | low/high, persona attribute |
| weight | float | Persona attribute (for future use) |
| age | int | Persona attribute (for future use) |
| baseline_steps_std | float | Persona attribute (for future use) |
| comb_scores | dict[theme→int] | COM-B survey (1-5 Likert), per-persona |

---

## 5. Action Space Design

### 12 Actions = 6 COM-B Themes × 2 Delivery Times

| Theme | COM-B Component | Morning (6 AM) | Afternoon (3 PM) |
|-------|-----------------|----------------|------------------|
| Ability | Capability (Psychological) | `ability_morning` | `ability_afternoon` |
| Perceived Benefit | Motivation (Reflective) | `perceived_benefit_morning` | `perceived_benefit_afternoon` |
| Planning | Motivation (Reflective) | `planning_morning` | `planning_afternoon` |
| Prioritization | Motivation (Reflective) | `prioritization_morning` | `prioritization_afternoon` |
| Social Opportunity | Opportunity (Social) | `social_opportunity_morning` | `social_opportunity_afternoon` |
| Physical Opportunity | Opportunity (Physical) | `physical_opportunity_morning` | `physical_opportunity_afternoon` |

Plus `idle` (control arm).

### Action Selection Flows

**Control arm:** Always `idle`. No notifications.

**Random arm:** Uniform random over 12 non-idle actions.

**Fixed arm:** COM-B barrier-score weighted multinomial (see §7).

**RL arm:** ε-greedy C-MAB over 12 non-idle actions. ε = 0.3 (our convention, maps to PEARL ε = 0.7).

---

## 6. Burden Mechanism

### Definition

**Burden = count of failed notifications in last 7 days.**

A notification "fails" when it does not cause deviation from baseline behavior.
The baseline is defined by the idle action's transition distribution:
`P(next_state | state, idle)`.

### P-Success Formula

For a given state `s` and action `a`:

```text
P_success(s, a) = 1 - Σ P(ns | s, a) * P(ns | s, idle)
```

This is the probability that independent samples from the action and idle transition
distributions differ. A value of 0 means the two distributions are identical; 1 means
they never produce the same outcome.

When multiple stochastic factors exist (PEARL format), P_success is computed per factor
and combined:

```text
P_success(s, a) = 1 - ∏_f [ Σ P_f(ns | s_f, a) * P_f(ns | s_f, idle) ]
```

### Implementation

1. **Precompute** at environment reset: for each (state, action) pair, compute
   `P_success(s, a)` from the transition table.
2. **After each non-idle step:** draw Bernoulli(P_success(s, a)).
   - Success → action "worked" → no burden
   - Failure → action "didn't work" → increment failure counter
3. **Rolling 7-day window** of failure counts.
4. **Map to burden level:** 0-2→low, 3-5→medium, 6+→high.

### Expected Burden Profiles

| Arm | Typical Success Rate | Burden Profile |
|-----|---------------------|----------------|
| RL | 50-70% | Low → Medium (if effective) |
| Random | 30-50% | Medium |
| Fixed | 40-60% | Medium |
| Control | N/A (always idle) | Always Low |

### Why 7-day window?

- Burden carries across days (not reset daily like sprint1)
- Naturally decays as old failures fall out of window
- PEARL peak decline observed within 14-21 days (constitution T3.1)
- 7-day window allows burden to accumulate and decay within this timeframe

---

## 7. Agent Design

### ComBWeightedFixedAgent (extends FixedAgent)

Extends `FixedAgent` to implement PEARL's Fixed arm logic.

**Logic:**
1. Read COM-B Likert scores from persona (loaded from synthetic file)
2. Compute barrier score per theme: `barrier = 5 - likert_score`
3. Sample theme from multinomial distribution weighted by barrier scores
   - Higher barrier → higher probability of receiving that theme
4. Sample timing from binomial based on stated preference
   - 70% preferred time, 30% other (50/50 for no preference)
5. Return action string: `{theme}_{timing}`

**Config:**
```yaml
- type: comb_weighted_fixed
  persona_comb_file: "config/pearl/comb_scores.json"
  persona_name: "base"
  time_preference: "morning"  # morning, afternoon, no_preference
```

### EpsilonGreedyAgent (existing, unchanged)

PEARL uses ε-greedy C-MAB. Existing `EpsilonGreedyAgent` works as-is.

**Convention note:**
- PEARL defines ε = exploit probability (higher = less exploration)
- Our code defines ε = explore probability (higher = more exploration)
- PEARL ε = 0.7 → our ε = 0.3 (1 - 0.7 = 0.3)
- Documented in config comments

---

## 8. Reward Function Design

### Phase 2 Initial (random transitions)

For testing with random transitions, use a simple step-based reward:

```yaml
reward:
  variables:
    step_reward:
      source: state.recent_steps_mean
      mapping: {low: -1.0, moderate: 0.0, high: 1.0}
    action_penalty:
      source: action
      mapping: {idle: 0.0, ...all 12 non-idle: 0.05}
  formula: "step_reward - action_penalty"
```

This avoids needing baseline in the state. The reward engine's `state.*` source pattern
works as-is.

### Phase 2 Later (bootstrap transitions)

When bootstrap transitions are available with persona baselines, switch to:

```
reward = (steps_24h - B(M,W)) / B(M,W)
```

where `B(M,W)` = individual baseline from 30-day pre-study window, stratified by
morning (`M`) and weekday (`W`).

At that point, either:
- Add `baseline_steps_mean` to state (increases state space), or
- Modify reward engine to accept persona constants from config

---

## 9. COM-B Survey File

### Format: `config/pearl/comb_scores.json`

```json
{
  "base": {
    "ability": 3,
    "perceived_benefit": 2,
    "physical_opportunity": 4,
    "planning": 2,
    "prioritization": 3,
    "social_opportunity": 3,
    "time_preference": "morning"
  },
  "goal_driven": {
    "ability": 4,
    "perceived_benefit": 4,
    "physical_opportunity": 3,
    "planning": 3,
    "prioritization": 4,
    "social_opportunity": 2,
    "time_preference": "afternoon"
  },
  "social_responder": {
    "ability": 3,
    "perceived_benefit": 3,
    "physical_opportunity": 2,
    "planning": 2,
    "prioritization": 2,
    "social_opportunity": 5,
    "time_preference": "morning"
  },
  "stable_maintainer": {
    "ability": 4,
    "perceived_benefit": 3,
    "physical_opportunity": 3,
    "planning": 3,
    "prioritization": 3,
    "social_opportunity": 2,
    "time_preference": "no_preference"
  },
  "resistant": {
    "ability": 2,
    "perceived_benefit": 1,
    "physical_opportunity": 1,
    "planning": 1,
    "prioritization": 1,
    "social_opportunity": 1,
    "time_preference": "no_preference"
  }
}
```

### Barrier Score Calculation

For each persona:
- `barrier_score[theme] = 5 - likert_score[theme]`
- Higher barrier → higher probability of receiving that theme

Example (base persona):
- ability: barrier = 5-3 = 2, weight = 2/14 = 0.14
- perceived_benefit: barrier = 5-2 = 3, weight = 3/14 = 0.21
- physical_opportunity: barrier = 5-4 = 1, weight = 1/14 = 0.07
- planning: barrier = 5-2 = 3, weight = 3/14 = 0.21
- prioritization: barrier = 5-3 = 2, weight = 2/14 = 0.14
- social_opportunity: barrier = 5-3 = 2, weight = 2/14 = 0.14

Normalized weights sum to 1.0. Multinomial sampling from these weights.

---

## 10. Implementation Plan (Ordered Steps)

### Step 1: Fixed Agent Extension

**Files to modify:**
- `src/rl_health_interventions/agents/fixed.py` — add `ComBWeightedFixedAgent` class
- `src/rl_health_interventions/agents/__init__.py` — register new type

**Implementation:**
```python
class ComBWeightedFixedAgent(FixedAgent):
    def __init__(self, action=None, seed=None, actions=None,
                 comb_scores=None, time_preference="no_preference"):
        self._comb_scores = comb_scores or {}
        self._time_preference = time_preference
        self._rng = random.Random(seed)
        self._actions = actions

    def select_action(self, state) -> str:
        # Compute barrier scores
        barriers = {theme: 5 - score for theme, score in self._comb_scores.items()}
        # Normalize to probabilities
        total = sum(barriers.values())
        probs = [barriers[theme] / total for theme in themes]
        # Sample theme
        theme = self._rng.choices(themes, weights=probs)[0]
        # Sample timing
        timing = self._sample_timing()
        return f"{theme}_{timing}"
```

**Test:** `tests/agents/test_fixed.py` — add test for COM-B weighted selection.

### Step 2: COM-B Survey File

**Files to create:**
- `config/pearl/comb_scores.json` — synthetic COM-B survey scores per persona

### Step 3: Burden Calculation in Environment

**Files to modify:**
- `src/rl_health_interventions/environment.py` — add P-success calculation

**Implementation:**
- New method `_precompute_success_probs(transition_table)` called at `__init__`
- For each (state, action) pair, compute `P_success(s, a)`
- New method `_update_burden(state, action, success)` called after each step
- Track failure count in rolling 7-day window
- Map to burden level and update state

**Modification to step():**
```python
if action != "idle":
    success = self._rng.random() < self._success_probs[state_key][action]
    if not success:
        self._failure_history.append(True)
    else:
        self._failure_history.append(False)
    # Trim to window size
    if len(self._failure_history) > 7:
        self._failure_history.pop(0)
    failure_count = sum(self._failure_history)
    burden = self._burden_mapping[failure_count]
    state = state.with_factors(burden=burden)
```

### Step 4: PEARL Config Files

**Files to create:**
- `docs/experimental_phases/pearl/pearl_random/configs/pearl_random.yaml`
- `docs/experimental_phases/pearl/pearl_bootstrap/configs/pearl_bootstrap.yaml`

**Config structure:** See §11.

### Step 5: Experiment Infrastructure

**Files to create:**
- `docs/experimental_phases/pearl/_shared.py`
- `docs/experimental_phases/pearl/run_experiments.py`
- `docs/experimental_phases/pearl/pearl_random/results/README.md`
- `docs/experimental_phases/pearl/pearl_bootstrap/results/README.md`

### Step 6: README.md

**File to create:**
- `docs/experimental_phases/pearl/README.md`

Contains:
- Experiment overview
- Feature importance ADR (D15 content)
- Agent descriptions
- Config reference
- Running instructions

### Step 7: Decision Catalogue Update

**File to modify:**
- `docs/research/decision-catalogue.md` — add D15 entry

### Step 8: Constitution Corrections

**File to modify:**
- `docs/research/pearl-constitution.md` — fix arm mappings, baseline period, T2.3 ordering

**Corrections needed:**
- T1.1/T2.1: 7-day baseline → 30-day pre-study window
- T2.2: remove "150-450" range, cite observed values (210-296)
- T2.3: RL ≥ Fixed ≥ Random > Control → RL > (Fixed ≈ Random ≈ Control)
- Arm mapping: Fixed = COM-B weighted, RL = ε-greedy (not Thompson Sampling)
- Constitution date: note corrections from Phase 1 deep analysis

---

## 11. Config File Structure

### `pearl_random.yaml`

```yaml
# PEARL-matched config with RANDOM transitions (for testing).
# 1 decision point/day, 12 actions, ε-greedy C-MAB.
# Burden: P-success (7-day window).
# Feature selection: weight >= 0.5 boundary (Figure 12, Lee 2025).
# Epsilon convention: PEARL ε=0.7 maps to our ε=0.3 (inverted convention).

initial_state:
  recent_steps_mean: moderate
  recent_walk_pattern: low
  morning_steps_ratio: balanced
  day_of_week: weekday
  burden: low

state:
  variables:
    recent_steps_mean:
      names: [low, moderate, high]
    recent_walk_pattern:
      names: [low, high]
    morning_steps_ratio:
      names: [morning, balanced, evening]
    day_of_week:
      names: [weekday, weekend]
      advanced:
        type: cyclic
        granularity: daily
        pattern: [weekday, weekday, weekday, weekday, weekday, weekend, weekend]
    burden:
      names: [low, medium, high]

actions:
  # 12 actions = 6 COM-B themes × 2 delivery times
  ability_morning: {}
  ability_afternoon: {}
  perceived_benefit_morning: {}
  perceived_benefit_afternoon: {}
  planning_morning: {}
  planning_afternoon: {}
  prioritization_morning: {}
  prioritization_afternoon: {}
  social_opportunity_morning: {}
  social_opportunity_afternoon: {}
  physical_opportunity_morning: {}
  physical_opportunity_afternoon: {}

reward:
  variables:
    step_reward:
      source: state.recent_steps_mean
      mapping: {low: -1.0, moderate: 0.0, high: 1.0}
    action_penalty:
      source: action
      mapping:
        idle: 0.0
        ability_morning: 0.05
        ability_afternoon: 0.05
        perceived_benefit_morning: 0.05
        perceived_benefit_afternoon: 0.05
        planning_morning: 0.05
        planning_afternoon: 0.05
        prioritization_morning: 0.05
        prioritization_afternoon: 0.05
        social_opportunity_morning: 0.05
        social_opportunity_afternoon: 0.05
        physical_opportunity_morning: 0.05
        physical_opportunity_afternoon: 0.05
  formula: "step_reward - action_penalty"

transition_model:
  type: random

episode_days: 60
steps_per_day: 1
seed: 42

agents:
  # Agent 0 — Control arm: always idle
  - type: fixed
    action: idle

  # Agent 1 — Random arm: uniform random over 12 actions
  - type: random

  # Agent 2 — Fixed arm: COM-B barrier-score weighted multinomial
  - type: comb_weighted_fixed
    persona_comb_file: "../../../../config/pearl/comb_scores.json"
    persona_name: "base"
    time_preference: "morning"

  # Agent 3 — RL arm: ε-greedy C-MAB
  - type: epsilon_greedy
    epsilon: 0.3  # PEARL ε=0.7 maps to our ε=0.3
```

### `pearl_bootstrap.yaml`

Same as `pearl_random.yaml` except:
```yaml
transition_model:
  type: bootstrap
  table_dir: ../../../../tables/persona/base_deepseek-v4-flash
```

When transition tables are regenerated for the 12-action space, this config will use them.

---

## 12. Verification Plan

### Phase 2 Tests

| Check | Method | Expected |
|-------|--------|----------|
| Config validation | `uv run python -c "from rl_health_interventions.config.loader import load_config; ..."` | No validation errors |
| Fixed agent COM-B weighted | Unit test | Barrier-weighted theme selection matches expected distribution |
| Burden P-success | Unit test | Given known transition table, burden computed correctly |
| Environment runs | `uv run pytest` | All existing tests pass |
| End-to-end episode | Run 1 episode per agent | No crashes, valid states |
| 4-arm comparison | Run 50 seeds, compare arm means | RL > others |

### Future Checks (after bootstrap tables)

| Check | Method | Constitution Tier |
|-------|--------|-------------------|
| Baseline stability | `scripts/pearl_constitution/run_baseline_check.py` | T1 |
| Effect size magnitude | `scripts/pearl_constitution/run_distribution_check.py` | T2 |
| Burden saturation | `scripts/pearl_constitution/run_behaviour_check.py` | T3 |
| Stress tests | `scripts/pearl_constitution/run_stress_tests.py` | T4 |

---

## 13. Risks and Open Questions

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Reward formula too simple for bootstrap phase | Medium | Use simple formula for random testing; document relative-change formula for bootstrap |
| 108 states may be too many for LLM bootstrapping | Low | 108 is manageable; sprint1 bootstrap uses similar scale |
| Transition table generation for 12 actions | High | Deferred to separate issue; use random for testing |
| Burden calculation depends on transition table quality | Medium | Random transitions produce reliable distributions; precompute once |
| COM-B survey is LLM-generated, not real data | Low | Document as synthetic; tune if unrealistic |

### Open Questions

1. **Persona baseline values**: Should `baseline_steps_mean` values be derived from PEARL population statistics (Table 2 vs Table 3 discrepancy)?

2. **Burden mapping thresholds**: 0-2/3-5/6+ are derived from effect size best guesses. Should these be sensitivity-tested?

3. **Epsilon value**: Is ε=0.3 right for PEARL (maps to ε=0.7)? PEARL also used ε=0.8 (our ε=0.2) for high-variance ensemble models.

4. **COM-B scores**: Synthetic scores are best guesses. Should scores be tuned per persona to produce realistic theme distributions?

---

## 14. File Manifest

### New Files

| File | Purpose |
|------|---------|
| `docs/experimental_phases/pearl/README.md` | Experiment overview + feature importance ADR |
| `docs/experimental_phases/pearl/_shared.py` | Config resolution, agent labels |
| `docs/experimental_phases/pearl/run_experiments.py` | Benchmark runner |
| `docs/experimental_phases/pearl/pearl_random/configs/pearl_random.yaml` | Random-transition config |
| `docs/experimental_phases/pearl/pearl_random/results/README.md` | Output description |
| `docs/experimental_phases/pearl/pearl_bootstrap/configs/pearl_bootstrap.yaml` | Bootstrap config (placeholder) |
| `docs/experimental_phases/pearl/pearl_bootstrap/results/README.md` | Output description |
| `config/pearl/comb_scores.json` | Synthetic COM-B survey scores |

### Modified Files

| File | Change |
|------|--------|
| `src/rl_health_interventions/agents/fixed.py` | Add `ComBWeightedFixedAgent` |
| `src/rl_health_interventions/agents/__init__.py` | Register `comb_weighted_fixed` type |
| `src/rl_health_interventions/environment.py` | Add P-success burden calculation |
| `docs/research/decision-catalogue.md` | Add D15 (feature selection) |
| `docs/research/pearl-constitution.md` | Fix arm mappings, baseline period, T2.3 |

---

## References

- Lee et al. (2025). "A Personalized Exercise Assistant using Reinforcement Learning (PEARL)." arXiv:2508.10060.
- Phase 1 deep analysis: `docs/research/recreations/pearl-rct-2025/pearl-deep-analysis.md`
- Decision catalogue: `docs/research/decision-catalogue.md`
- Constitution: `docs/research/pearl-constitution.md`
- Existing config: `config/pearl_constitution.yaml`
- Issue #252: [PEARL-matched config](https://github.com/wd7512/rl-health-interventions/issues/252)

---

## 15. Fix Plan — Addressing Implementation Gaps

**Date:** 2026-07-24
**Author:** opencode (post-implementation review)

The copilot implementation shipped the skeleton (agent, config, tests, runner) correctly but
skipped or substituted several design-critical items. This section defines the fix plan.

### 15.1 Flaw Inventory

| # | Flaw | Category | Effort |
|---|------|----------|--------|
| 1 | P-success burden replaced with naive rolling_window_count action counter | Core design | Large |
| 2 | No 12-action transition tables exist (blocks #1) | Infrastructure | Medium |
| 3 | `BootstrapTransition._build_state_key` hardcodes sprint1 factor names (`step_bin`, `burden`, `day_of_week`, `sleep`) — can't load PEARL tables with different stochastic factors | Design bug | Small |
| 4 | No D15 decision catalogue entry (feature selection ADR from Figure 12) | Documentation | Small |
| 5 | Constitution corrections not applied (arm mappings, baseline period, T2.3 ordering) | Documentation | Small |
| 6 | No README for pearl_random experiment | Documentation | Small |
| 7 | No bootstrap config variant (`pearl_bootstrap.yaml`) | Config | Small |
| 8 | Degenerate reward results not explained anywhere (Control wins because actions carry penalty with no upside under random transitions) | Documentation | Small |

### 15.2 Part A: Transition Infrastructure (unblocks #1, #2)

#### A1. New `RandomTransitionSA` transition model

**New file:** `src/rl_health_interventions/transitions/random_sa.py`

- Extends `TransitionModel`
- At `__init__`, for each stochastic factor AND each (state_value, action) pair, draw a
  **separate** Dirichlet(1.0) sample — unlike `RandomTransition` which uses one shared
  Dirichlet draw per factor regardless of (state, action)
- Internal storage: `self._tables[step_idx][factor_name]["{state_val}|{action}"] = (targets, probs)`
- `transition(state, action)` looks up the appropriate table entry for the current
  (state_value, action) pair
- `day_boundary.json` key: `{factor1_val}|{factor2_val}|...` (stochastic factors only, no action)
- `within_day_N.json` key: `{factor1_val}|{factor2_val}|...|{action}` (stochastic factors + action)
- `save_tables(output_dir)` serializes to bootstrap-compatible JSON format

State space math for PEARL 12-action:
- 3 stochastic factors (`recent_steps_mean`, `recent_walk_pattern`, `morning_steps_ratio`)
  with 3×2×3 = 18 unique factor combos
- 18 combos × 13 actions = 234 within_day entries per step
- 18 day_boundary entries
- Trivial

**Register** in `transitions/__init__.py` and add `"random_sa"` to `_KNOWN_TRANSITION_TYPES`
in `config/schemas.py`.

#### A2. Table generation script

**New file:** `scripts/generate_pearl_tables.py`

- Loads `pearl_random.yaml` config
- Instantiates `RandomTransitionSA` with a seed
- Calls `save_tables("tables/pearl_12action/")`
- Produces `day_boundary.json` + `within_day_0.json`
- Run once, commit output

#### A3. Generalize `BootstrapTransition._build_state_key`

**Modify:** `src/rl_health_interventions/transitions/bootstrap.py`

Currently hardcodes factor names from sprint1. Change to derive from config's stochastic
factors and advanced factors that influence transitions:

```python
def _build_state_key(self, state, action, *, for_within_day):
    factors = state.factor_values
    parts = [factors[name] for name in self._stochastic_factors]
    if for_within_day:
        parts.append(action)
    # Include advanced factors that affect transitions (e.g., day_of_week)
    for name in self._config.state.variables:
        if name not in self._stochastic_factors and name in factors:
            parts.append(factors[name])
    return "|".join(parts)
```

This makes `BootstrapTransition` config-agnostic — works with both sprint1 (4-action) and
PEARL (12-action) configs.

### 15.3 Part B: P-Success Burden

#### B1. Add `_precompute_p_success()` to Environment

**Modify:** `src/rl_health_interventions/environment.py`

After creating the transition model, check if it exposes `day_boundary` / `within_day`
tables (both `BootstrapTransition` and `RandomTransitionSA` will). If so, precompute:

```python
P_success(s, a) = Σ_ns P(ns | s, a)² / (P(ns | s, a) + P(ns | s, idle))
```

For each (state_key, action) pair, where `P(ns | s, a)` is looked up from the action's
table row and `P(ns | s, idle)` from idle's table row for the same state.

#### B2. Modify `_apply_rolling_advances` for P-success burden

When `_p_success` is populated and the factor being computed is `burden`:
1. Look up `p_success` for current state + action from precomputed table
2. Draw Bernoulli: `success = rng.random() < p_success`
3. If not success → count as failure in the rolling window
4. Map failure count to burden level via existing mapping (0-2→low, 3-5→medium, 6+→high)

When `_p_success` is empty (no tables available), fall back to current naive counting
(action count in window).

#### B3. Add RNG to Environment

`Environment.__init__` needs `self._rng = np.random.default_rng(seed)` for the Bernoulli
draw. The seed is already passed as a parameter.

### 15.4 Part C: Config Changes

#### C1. Update `pearl_random.yaml`

- Change `transition_model.type` from `random` to `random_sa`
- Add `table_dir: tables/pearl_12action` (for table serialization target)
- Add header comments explaining:
  - `random_sa` generates per-(state,action) tables enabling P-success
  - P-success burden makes non-idle actions non-trivially different from idle
  - Under random transitions, Control winning is expected (degenerate reward)
  - Relative-change reward formula deferred to bootstrap phase

#### C2. Create `pearl_bootstrap/configs/pearl_bootstrap.yaml`

Copy of pearl_random.yaml with:
- `transition_model.type: bootstrap`
- `transition_model.table_dir: tables/pearl_12action` (same generated tables)
- Comments noting this is the "real" experiment config once tables are committed

### 15.5 Part D: Documentation

#### D1. D15 entry in `decision-catalogue.md`

Feature selection ADR from Figure 12 (Lee 2025):
- Status: closed
- Boundary: weight >= 0.5 (natural break, next is 0.32 = 47% drop)
- 8 features: 5 static (baseline_steps_mean, weight, age, baseline_walk_pattern,
  baseline_steps_std) + 3 dynamic (recent_steps_mean, recent_walk_pattern,
  morning_steps_ratio)
- Evidence: XGBoost model weights, 7,711 participants, 60 days
- Sub-questions: sensitivity analysis on threshold (0.4 vs 0.5 vs 0.6)

#### D2. Constitution corrections in `pearl-constitution.md`

| Section | Current (wrong) | Corrected |
|---------|-----------------|-----------|
| Arm table: Fixed | `FixedAgent(action="movement_suggestion")` | `ComBWeightedFixedAgent` — COM-B barrier-weighted multinomial |
| Arm table: RL | `ThompsonSampling` | `EpsilonGreedyAgent(epsilon=0.3)` — ε-greedy C-MAB |
| T1.1 | "7-day baseline period" | "30-day pre-study window" |
| T2.1 | "7-day baseline" | "30-day pre-study window" |
| T2.3 | `RL ≥ Fixed ≥ Random > Control` | `RL > (Fixed ≈ Random ≈ Control)` |

#### D3. README for pearl_random experiment

**New file:** `docs/experimental_phases/pearl_random/README.md`

Contents:
- Experiment overview (PEARL 4-arm comparison, random transitions)
- Agent descriptions (Control, Random, Fixed COM-B, RL ε-greedy)
- State space (108 states, 13 actions)
- How to run
- Known limitations:
  - Random transitions → no causal link between actions and outcomes
  - Optimal policy is "always idle" — Control winning is expected
  - P-success burden makes actions non-trivially different from idle
  - Relative-change reward formula deferred to bootstrap phase

### 15.6 Part E: Tests

#### E1. `RandomTransitionSA` unit tests

**New file:** `tests/unit/transitions/test_random_sa.py`

- Test that different (state, action) pairs produce different probability vectors
- Test `save_tables` produces valid bootstrap-compatible JSON
- Test round-trip: save → load via `BootstrapTransition` → same samples
- Test key format matches expected `{factor1}|{factor2}|...|{action}` pattern

#### E2. P-success burden tests

- Test that with `_p_success` populated, non-idle actions can produce `burden=medium` or
  `burden=high`
- Test that idle actions never increment failure count
- Test that `_precompute_p_success` produces valid probabilities (0 < p < 1)

#### E3. Update regression test

- Add regression for pearl_random with `random_sa` transitions + P-success burden
- Regenerate golden fixtures (results will differ from current)

### 15.7 Execution Order

```
A1 (RandomTransitionSA)  ──┐
A3 (Generalize Bootstrap) ──┤
                            ├── A2 (Generate tables) ── B1-B3 (P-success burden) ── C1-C2 (Configs)
                            │                                                          │
D1-D3 (Docs) ──────────────┤                                                          │
E1-E3 (Tests) ─────────────┴──────────────────────────────────────────────────────────┘
```

Part A is the foundation — everything else depends on having per-(state,action) transition
tables. Parts D (docs) can be done in parallel with Part A since they're independent.

### 15.8 Verification After Fix

| Check | Method | Expected |
|-------|--------|----------|
| RandomTransitionSA tables valid | Unit test | All keys present, probs sum to 1.0 |
| BootstrapTransition loads PEARL tables | Unit test | No hardcoded factor name errors |
| P-success precomputed | Unit test | 0 < P_success < 1 for all (s, a) |
| Burden varies across arms | End-to-end | Control=always low, others=low/medium/high mix |
| D15 in catalogue | Read check | Entry exists with evidence citations |
| Constitution corrected | Read check | Arms, baseline, T2.3 match PEARL paper |
| All tests pass | `uv run pytest` | No regressions |
| Lint clean | `uv run ruff check` | No new warnings |

### 15.9 Post-Implementation: Consolidation

**Decision:** `RandomTransitionSA` was removed and consolidated into `table_transition`.

Rationale:
- `RandomTransitionSA` generated tables in-memory at init; `TableTransition` loads
  pre-generated tables from disk. At runtime they produce identical behavior.
- Table generation is a one-time setup step, not a runtime concern.
- `scripts/generate_pearl_tables.py` moved to
  `docs/experimental_phases/pearl_random/generate_tables.py`.

Also renamed `bootstrap` → `table_transition` for clarity: the type reads
pre-computed transition tables from disk — "bootstrap" was a misleading name
(from the LLM bootstrapping pipeline that originally generated sprint1 tables).

Files changed:
- Deleted: `transitions/random_sa.py`, `tests/unit/transitions/test_random_sa_transition.py`,
  `scripts/generate_pearl_tables.py`
- Renamed: `transitions/bootstrap.py` → `transitions/table_transition.py`
- Renamed class: `BootstrapTransition` → `TableTransition`
- Renamed registry key: `"bootstrap"` → `"table_transition"`
- All 35 YAML configs updated: `type: bootstrap` → `type: table_transition`
