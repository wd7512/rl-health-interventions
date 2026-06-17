# rl-health-interventions

A config-driven RL simulation framework for testing health interventions. Define MDP, agents, and experiments in YAML — no code changes needed.

## What this is

A config-driven RL simulation framework for testing health interventions. Define MDP, agents, and experiments in YAML — no code changes needed. The MVP uses a binary state/action MDP (nudge/idle, sedentary/active) with configurable transition probabilities and per-step reward multipliers. Thompson Sampling, epsilon-greedy, UCB, and random agents on a 90-day × 5 steps/day episode. See `docs/design/initial_design.tex` for the long-term vision.

## Quickstart

```bash
uv sync --dev

# Run with default config (Thompson Sampling, 450 steps)
uv run rl-health-interventions --config config/rule_based.yaml --output results.csv

# Run with epsilon-greedy baseline
uv run rl-health-interventions --config config/rule_based.yaml --agent epsilon_greedy --output results.csv

# Run with random baseline
uv run rl-health-interventions --config config/rule_based.yaml --agent random --output results.csv

# Run with UCB1 baseline
uv run rl-health-interventions --config config/rule_based.yaml --agent ucb --output results.csv
```

## How it works

The MDP is defined entirely in YAML:

```yaml
# config/rule_based.yaml (simplified)
states:
  sedentary:
    reward: 0.0
  active:
    reward: 1.0
actions: [nudge, idle]
steps_per_day: 5
episode_days: 90
transition_model:
  type: rule_based
  transition_probabilities:
    sedentary:
      nudge: {active: 0.3, sedentary: 0.7}
      idle: {active: 0.1, sedentary: 0.9}
    active:
      nudge: {active: 0.5, sedentary: 0.5}
      idle: {active: 0.6, sedentary: 0.4}
# Optional: reward_multiplier_by_step: [1, 1, 1, 1, 0]
```

Change the YAML to change the experiment. No code changes needed.

## Agents

| Agent | Type | Learning | CLI flag |
|-------|------|----------|----------|
| Thompson Sampling | Beta-Bernoulli bandit | Yes (posterior update) | `--agent thompson_sampling` |
| Epsilon-Greedy | Q-learning bandit | Yes (incremental average) | `--agent epsilon_greedy` |
| UCB1 | Upper Confidence Bound | Yes (confidence interval) | `--agent ucb` |
| Random | Uniform random | No | `--agent random` |

All agents are bandits in the MVP — they do not condition on state. State-aware agents are planned for Phase 2.

## MDP formulation

See `docs/mvp/mvp_specification.tex` and `docs/design/initial_design.tex` for the full MDP formulation.

**Initial results** (config/rule_based.yaml, 10 seeds, 450 timesteps each):

| Agent | Total Reward | Mean Reward/Step |
|-------|-------------|-----------------|
| Thompson Sampling | 160.3 ± 10.5 | 0.356 ± 0.023 |
| Epsilon-Greedy (ε=0.1) | 160.8 ± 15.3 | 0.357 ± 0.034 |
| UCB (c=2.0) | 148.2 ± 15.0 | 0.329 ± 0.033 |
| Random | 135.6 ± 10.7 | 0.301 ± 0.024 |

## Project structure

```
src/rl_health_interventions/
  config/        # MDPConfig, AgentConfig, YAML loader
  agents/        # Thompson Sampling, epsilon-greedy, UCB, random
  transitions/   # RuleBasedTransition (config-driven matrix)
  rewards/       # CompoundReward (precomputed per-step reward)
  state.py       # StateView dataclass
  environment.py # step/reset simulation loop
  experiment.py  # run_episode + run_experiment
config/          # YAML config files
docs/mvp/        # MDP formulation + results
tests/           # 135 tests (unit + integration)
```

## Development

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
uv build
```

## Non-goals (current phase)

Per `docs/plans/ROADMAP.md`, these are deferred until after the MVP ships:

- Multi-timescale reward (immediate + delayed body measure)
- 4 user archetypes (goal-driven, social, resistant, stable)
- Burden accumulation / decay model
- Evaluation framework (bootstrap CIs, power analysis)
- Multi-feature synthetic data matched to population statistics
- Safety / ethics review

## References

- HeartSteps V2 (Klasnja et al., 2022)
- `docs/design/initial_design.tex` (MDP formalisation)
- `docs/plans/ROADMAP.md` (backlog, rough guidance)
