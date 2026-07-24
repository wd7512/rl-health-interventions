# PEARL Random Transitions Experiment

## Overview

PEARL 4-arm comparison using per-(state,action) random transitions. This is the
**testing/validation** config — random transitions mean there is no causal link
between actions and state outcomes. The optimal policy under random transitions
is "always idle" (no action penalty, same expected outcome).

## Agents

| Arm | Agent | Description |
|-----|-------|-------------|
| Control | `FixedAgent(action="idle")` | Always idle — no action penalty |
| Random | `RandomAgent` | Uniform random over all 13 actions |
| Fixed | `ComBWeightedFixedAgent` | COM-B barrier-weighted multinomial (persona: base) |
| RL | `EpsilonGreedyAgent(epsilon=0.3)` | ε-greedy C-MAB, learns action values |

## State Space

- 3 stochastic factors: `recent_steps_mean` (3), `recent_walk_pattern` (2), `morning_steps_ratio` (3)
- 2 advanced factors: `day_of_week` (cyclic), `burden` (rolling window count)
- Total states: 3 × 2 × 3 × 2 × 3 = 108

## Action Space

13 actions: `idle` + 12 COM-B interventions (6 themes × 2 times-of-day)

## Transition Model

`random_sa`: Per-(factor_value, action) Dirichlet tables. Enables Bayesian
P-success burden computation — non-idle actions have P(success) < 1.0,
so failures accumulate and burden increases over time.

## Burden Mechanism

Bayesian P-success: `P(success | s, a) = 1 - Σ_t P(t|s,a) * P(t|s,idle)`

- Idle actions never count as failure (P_success = 1.0)
- Non-idle actions have P_success ∈ (0, 1) based on how much their
  transition distribution differs from idle
- Failure count in rolling window maps to burden level:
  0-2 failures → low, 3-5 → medium, 6+ → high

## Known Limitations

- Random transitions → no causal link between actions and outcomes
- Optimal policy is "always idle" — Control winning is expected
- Bayesian P-success burden makes actions non-trivially different from idle
- Relative-change reward formula deferred to bootstrap phase
- Reward formula: `step_reward - action_penalty` (action penalty penalizes all non-idle actions)

## Running

```bash
uv run python docs/experimental_phases/pearl_random/run_experiments.py
```
