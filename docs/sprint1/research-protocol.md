---
title: "Research Protocol — Sprint 1: RL-Driven Physical Activity Interventions"
status: "draft"
date: "2026-06-28"
supersedes: "docs/design/initial_design.tex (Sprint 1 scope only)"
---

# Research Protocol — Sprint 1

## 1. Project Overview

**rl-health-interventions** is a configurable simulation framework for evaluating reinforcement learning (RL) agents that personalise just-in-time adaptive interventions (JITAIs) for physical activity. The framework is built around a YAML-driven architecture where the MDP specification, transition dynamics, reward function, and agent configurations are all defined in config files — no code changes required to run different experiments.

This document specifies the Sprint 1 research protocol. Sprint 1 delivers a fully functional MDP environment with a 36-state factored state space, 4 actions, LLM-bootstrapped transition tables, configurable reward, and a suite of model-free agents (contextual bandits + optional Q-learning). The environment produces reproducible evaluation results across 50 random seeds for rigorous statistical comparison.

### 1.1 Research Objectives

1. Evaluate whether contextual bandit algorithms can learn effective intervention policies in a 36-state, 4-action MDP with LLM-simulated user transitions, measured against random, idle-only, and fixed-ratio baselines.

2. Validate the factored transition architecture by comparing within-day and day-boundary transition dynamics across different burden levels, sleep states, day types, and action choices.

3. Establish a reproducible evaluation pipeline where the same config + seed produces identical results across platforms, enabling downstream researchers to compare new algorithms against a published benchmark.

4. Demonstrate the feasibility of LLM-bootstrapped transition models as a low-cost alternative to hand-crafted transition matrices for health intervention simulations.

### 1.2 Relation to the Design Document

The complete project specification is in `docs/design/initial_design.tex`. Sprint 1 implements the subset that can be built with synthetic (LLM-generated) transitions. Deferred features:

| Feature | Phase | Reason |
|---------|-------|--------|
| 4 user archetypes | Phase 2 | Requires real-data calibration |
| User profiles (age/gender/baseline) | Phase 2 | Not needed for single-context evaluation |
| Non-activity reward (journal proxy) | Phase 2 | Novel mechanism |
| Multi-objective reward (steps + mood) | Phase 2 | Requires multi-timescale data |
| Trend dimension | Phase 2 | No precedent |
| Mood/stress state variables | Phase 2 | Null evidence in existing MRTs |
| Delayed reward (3-week body measures) | Phase 2 | Sparse reward complicates learning |
| Deep RL (DQN, PPO) | Phase 3 | Higher sample complexity |
| Real data calibration | Phase 4 | Requires data access applications |
| Safety constraints, ethics | Phase 4 | Required before real-data deployment |

## 2. MDP Formalisation

The MDP is defined as the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$.

### 2.1 Assumptions

- **Full observability:** The agent observes the complete state at each decision point.
- **Fixed episode length:** 90 days × 5 steps/day = 450 decision points.
- **No discounting:** $\gamma = 1.0$ for bandit agents; $\gamma = 0.99$ for Q-learning.
- **Immediate reward horizon:** Each step reflects activity in the 30-minute window after the decision (HeartSteps V2 convention).
- **Single blank context:** No archetypes. All users share one transition model.

### 2.2 State Space $\mathcal{S}$

$$
s_t = (\text{step\_bin}, \text{sleep}, \text{day\_of\_week}, \text{burden})
$$

| Dimension | Levels | Values | Cardinality | Evidence |
|-----------|--------|--------|-------------|----------|
| `step_bin` | 3 | inactive / moderate / active | 3 | Lee 2022 (JAMA): minimal effective dose ~4k for mortality benefit. Saint-Maurice 2020 (JAMA): mortality RR plateaus at 6-8k. Paluch 2022 (Lancet PH): 15% RR reduction per 1k steps. |
| `sleep` | 2 | rested (>7h) / under-rested (≤7h) | 2 | Irwin 2016 (Lancet): sleep interventions have moderate effect (g=0.43). |
| `day_of_week` | 2 | weekday / weekend | 2 | Universal across all 6 reference systems. |
| `burden` | 3 | low (0) / medium (1) / high (2-3) | 3 | StepCountJITAI: burden-aware > burden-agnostic. Rolling window has zero tuning parameters. |

**Total:** $3 \times 2 \times 2 \times 3 = 36$ states.

#### Step Bin Boundaries (per-timestep)

| Bin | Label | Per-timestep | Daily |
|-----|-------|-------------|-------|
| 0 | inactive | <800 | <4,000 |
| 1 | moderate | 800–1,600 | 4,000–8,000 |
| 2 | active | >1,600 | >8,000 |

The LLM outputs raw step counts; the environment accumulates into daily total and bins.

#### Deterministic Dimensions

| Dimension | Update Rule |
|-----------|------------|
| `day_of_week` | Advances 1/day (0→...→6→0). Weekday (0-4) / weekend (5-6). |
| `burden` | Rolling count of non-idle actions in last 3 timesteps. Crosses day boundaries. Early episode: incomplete window → low. |

### 2.3 Action Space $\mathcal{A}$

| Action | Penalty | Burden | Description |
|--------|---------|--------|-------------|
| `idle` | 0 | 0 | No notification sent |
| `movement_suggestion` | -1 | +1 | Recommend a short walk (HeartSteps mapping) |
| `goal_reminder` | -1 | +1 | Remind of step goal (Trella 2022 distinct construct) |
| `journal` | -1 | +1 | Expressive writing placeholder; no step benefit in Sprint 1 |

All non-idle actions carry the same burden impact (count-based, not severity-weighted). This aligns with D10 (rolling window with zero tuning parameters).

### 2.4 Transition Model $P$

**Within-day (steps 1-4):** $P(\text{step\_bin}' \mid \text{step\_bin}, \text{sleep}, \text{day\_of\_week}, \text{burden}, \text{action}, \text{time\_slot})$

**Day-boundary (step 0):** $P(\text{sleep}' \mid \text{step\_bin}, \text{sleep}, \text{day\_of\_week}, \text{burden}, \text{action})$

Factored into 6 tables:

| Table | Pairs | Cells | LLM calls |
|-------|-------|-------|-----------|
| Day-boundary | 144 | 288 | 2,880 |
| Within-day × 5 | 720 | 2,160 | 21,600 |
| **Total** | **864** | **2,448** | **24,480** |

**Day execution order:**

| Step | Table | Input sleep | Outputs |
|------|-------|-------------|---------|
| 0 | Day-boundary → Within-day #0 | sleep → sleep' | sleep', step_bin' |
| 1 | Within-day #1 | sleep' | step_bin' |
| 2 | Within-day #2 | sleep' | step_bin' |
| 3 | Within-day #3 | sleep' | step_bin' |
| 4 | Within-day #4 | sleep' | step_bin' |

Sleep is sampled once at the day boundary and remains constant for all 5 within-day timesteps. The day-boundary table uses end-of-day step bin context.

#### 2.4.1 LLM Bootstrapping

**Model:** DeepSeek V4 Flash (OpenRouter, $0.09/$0.18 per 1M tokens). **Calls:** 24,480. **Total output cells:** 2,448 (10 samples/cell). **Est. cost:** ~$0.15 with cache.

**System prompt** (cached once):
```
# Reference
5 timesteps per day: morning, mid-morning, lunch, afternoon, evening
Per-timestep step ranges (daily threshold / 5):
  <800 steps     = inactive
  800-1600 steps = moderate
  >1600 steps    = active
Sleep: >7h = well rested  |  ≤7h = under-rested
Burden (notification fatigue):
  low     = 0 of last 3 timesteps had an intervention
  medium  = 1 of last 3
  high    = 2 or 3 of last 3
```

**Within-day prompt (timestep 1 — morning):**
```
# Current state
You just woke up. It is the morning. It is a {weekday/weekend}.
You slept {>7h / ≤7h}. Your notification fatigue is {low/medium/high}.

{Your phone just said: {movement_suggestion / goal_reminder / journal}.

How many steps do you take this timestep?
```

**Within-day prompt (timesteps 2-5):**
```
# Current state
It is the {mid-morning / lunch / afternoon / evening}. Last timestep
({morning / mid-morning / lunch / afternoon}) you took {midpoint_step_count} steps.
Your notification fatigue is {low/medium/high}. It is a {weekday/weekend}.
You slept {>7h / ≤7h}.

{Your phone just said: {movement_suggestion / goal_reminder / journal}.
 -or- No action.}

How many steps do you take this timestep?
```

**Day-boundary prompt:**
```
# Current state
You are at the end of the day. It was a {weekday/weekend}.
You slept {>7h / ≤7h} last night. Your notification fatigue is {low/medium/high}.

Your day:
  morning:      {step_count} steps ({step_bin})
  mid-morning:  {step_count} steps ({step_bin})
  lunch:        {step_count} steps ({step_bin})
  afternoon:    {step_count} steps ({step_bin})
  evening:      {step_count} steps ({step_bin})

Your notifications today: {actions}

It's bedtime. Do you sleep >7 hours tonight?
{"sleep_>7h": true}
{"sleep_>7h": false}
```

#### 2.4.2 Random Transition Model (Dev Fallback)

`RandomTransition` generates uniform-Dirichlet-random probability tables matching the 6-table structure. All downstream code can be built and tested without the LLM pipeline.

### 2.5 Reward Function $R$

$$
R(s_t, a_t, s_{t+1}) = f(\text{step\_bin}') + \lambda \cdot \text{action\_penalty}(a_t)
$$

- $f(\text{inactive}) = 0.0$, $f(\text{moderate}) = 0.5$, $f(\text{active}) = 1.0$
- $\text{action\_penalty}(\text{idle}) = 0$, $\text{action\_penalty}(\text{non-idle}) = -1$
- $\lambda = 0.05$

Simplified: $R = \text{step\_bin\_reward} - 0.05 \cdot \mathbb{1}[\text{action} \neq \text{idle}]$

**Rationale:** Moderate activity (4-8k daily) is clinically meaningful (Lee 2022). λ = 0.05 makes one step_bin improvement (0.5) worth 10 interventions, preventing spamming. Burden handles saturation through reduced future activity probability; λ handles immediate interruption cost.

### 2.6 Agent Suite

| Agent | Type | Parameters |
|-------|------|------------|
| Random | Baseline | — |
| Idle-only | Baseline | — |
| Fixed-ratio | Baseline | Intervene every other step |
| Thompson Sampling | Contextual bandit | α=1.0, β=1.0 |
| Epsilon-Greedy | Contextual bandit | ε=0.1 |
| UCB | Contextual bandit | c=2.0 |
| Decaying Epsilon-Greedy | Contextual bandit | ε: 0.5 → 0.01 |
| Q-learning (optional) | Tabular RL | γ=0.99, α=0.1 |

All contextual bandits maintain independent parameters per (s, a) pair. Total: 144 parameter groups (36 states × 4 actions).

## 3. Config-Driven Architecture

The framework is YAML-driven. No code changes required to change the MDP specification.

### 3.1 Sprint 1 Config

```yaml
initial_state:
  step_bin: inactive
  sleep: rested
  day_of_week: weekday
  burden: low

state:
  step_bin:
    dims: 3
    names: [inactive, moderate, active]
    boundaries: [800, 1600]
  sleep:
    dims: 2
    names: [rested, under_rested]
  day_of_week:
    dims: 2
    names: [weekday, weekend]
  burden:
    dims: 3
    names: [low, medium, high]

actions:
  idle: {action_penalty: 0}
  movement_suggestion: {action_penalty: -1}
  goal_reminder: {action_penalty: -1}
  journal: {action_penalty: -1}

reward:
  step_bin: {inactive: 0.0, moderate: 0.5, active: 1.0}
  action_penalty_multiplier: 0.05

transition_model:
  type: bootstrap  # or "random" for dev
  table_dir: ../tables

episode_days: 90
steps_per_day: 5
seed: 42
```

### 3.2 Table Files

Six JSON files in `docs/sprint1/tables/`:

```
tables/
├── day_boundary.json
├── within_day_0.json   (morning)
├── within_day_1.json   (mid-morning)
├── within_day_2.json   (lunch)
├── within_day_3.json   (afternoon)
└── within_day_4.json   (evening)
```

Each maps composite (s, a) keys to probability vectors over the output dimension.

## 4. Implementation

### 4.1 Key Interfaces

**StateView (frozen dataclass):**
```
step_bin: int (0-2), sleep: int (0-1), day_of_week: int (0-1),
burden: int (0-2), day: int, step_of_day: int (0-4), steps_per_day: int = 5
  → state_key: tuple[int, int, int, int]  # hashable composite key
```

**TransitionModel ABC:**
```
transition(state: StateView, action: str) -> StateView
```

**RewardHandler ABC:**
```
reward(state: StateView, action: str, next_state: StateView, step_idx: int) -> float
```

### 4.2 Environment Step Logic

```
if step_of_day == 0:
    sleep' = sample day_boundary_table(current_state, action)
else:
    sleep' = current_state.sleep  # unchanged

step_bin' = sample within_day_table[step_of_day](current_state with sleep', action)
cumulative_steps += inferred_raw_steps
day_of_week' = (day_of_week + 1) % 7 if last step of day else day_of_week
burden' = rolling count of non-idle in last 3

next_state = StateView(step_bin', sleep', day_of_week', burden', ...)
reward = reward_handler.reward(current_state, action, next_state, step_of_day)
```

### 4.3 Module Structure

```
src/rl_health_interventions/
├── config/schemas.py          # Pydantic: MDPConfig, StateDimConfig, RewardConfig
├── config/loader.py           # YAML → Pydantic
├── state.py                   # StateView with state_key property
├── environment.py             # Environment with factored transition logic
├── transitions/
│   ├── _base.py               # TransitionModel ABC
│   ├── rule_based.py          # Existing 2-state model (unchanged)
│   ├── random.py              # RandomTransition (Dirichlet weights)
│   └── bootstrap.py           # BootstrappedTransition (JSON table loader)
├── rewards/
│   ├── _base.py               # RewardHandler ABC
│   ├── compound.py            # Existing per-step reward (unchanged)
│   └── sprint1.py             # Sprint1Reward: step_bin lookup + λ·penalty
├── agents/                    # Existing agents updated for state_key routing
├── episode.py                 # run_episode(): single-agent training loop
├── sweep.py                   # run_experiment(): multi-agent, multi-seed
└── __main__.py                # CLI entry point
```

### 4.4 Reproducibility

- Config specifies `seed: 42` (configurable). Each agent receives `derive_agent_seed(base, index)`.
- Same config + seed + tables = identical CSV output across platforms.
- Episode CSV: `step, day, step_of_day, state_key, action, reward`.
- Regression tests with JSON fixtures (exact match to 6dp).
- 50 seeds per configuration. Metrics: mean ± 95% bootstrap CI.

## 5. Evaluation Protocol

### 5.1 Metrics

| Metric | Definition | Purpose |
|--------|------------|---------|
| Total Reward | $\sum R_t$ over 450 steps | Primary comparison |
| Per-Step Reward | $\frac{1}{T} \sum R_t$ | Normalised |
| Last 50 Steps | $\frac{1}{50} \sum R_{400:449}$ | Convergence |
| % Gap vs Optimal | $100 \cdot (R^* - R) / R^*$ | Regret |

### 5.2 Protocol

- 50 seeds per configuration.
- Reporting: mean ± 95% bootstrap CI.
- Pairwise bootstrap test vs best baseline. Cohen's d effect size.
- Bonferroni correction: 4 agents × 3 baselines = 12 comparisons.

### 5.3 Verification

1. **Determinism:** Same config + seed = identical CSV.
2. **Baseline sanity:** idle-only < random in total reward (idle misses step improvements).
3. **Transition sanity:** high burden → lower P(step_bin' > step_bin) than low burden.
4. **Convergence:** Thompson Sampling ≥ 5% reward gain over random baseline.

## 6. Cost Estimates

| Model | Input / 1M | Output / 1M | Est. cost (75% cache) |
|-------|-----------|-------------|----------------------|
| DeepSeek V4 Flash | $0.09 | $0.18 | ~$0.15 |
| GLM 5.2 | $0.95 | $3.00 | ~$1.35 |
| Claude Sonnet 4.6 | $3.00 | $15.00 | ~$4.60 |
| Claude Opus 4.8 | $5.00 | $25.00 | ~$7.40 |
| GPT-5.5 | $5.00 | $30.00 | ~$7.80 |

24,480 calls × ~130-190 tokens/call = ~3.4M total tokens. DeepSeek V4 Flash recommended.

## 7. References

- Klasnja et al. 2019. HeartSteps V1. *Annals of Behavioral Medicine*.
- Liao et al. 2019. HeartSteps V2. *JMIR mHealth and uHealth*.
- Trella et al. 2022. Designing RL for Digital Interventions. *NPJ Digital Medicine*.
- Karine et al. 2024. StepCountJITAI. *JMIR Serious Games*.
- Lee et al. 2022. Step Counts and Mortality. *JAMA Internal Medicine*.
- Saint-Maurice et al. 2020. Daily Steps and Mortality. *JAMA*.
- Paluch et al. 2022. Steps and Health Outcomes. *The Lancet Public Health*.
- Gateno et al. 2023. Health Gym. *NeurIPS*.
- Smyth et al. 2018. Expressive Writing. *JMIR Mental Health*.
- Irwin et al. 2016. Sleep Interventions. *The Lancet*.

## 8. Dependencies

- Python 3.11+, numpy, pydantic, pyyaml, httpx (LLM script only), pytest, ruff, ty. Managed via uv.

## 9. Decision Log

See `docs/research/resolved-decisions-sprint-1.md` for all 14 decisions (D1–D14) with evidence tables and rationale. Deferred decisions in `docs/research/future-sprints.md`.

## 10. Project Structure

See `docs/sprint1/configs/sprint1.yaml` for the locked-in config. See `src/rl_health_interventions/` for implementation. See `docs/research/` for evidence and decision records.
