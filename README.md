# rl-health-interventions

A config-driven simulation framework for testing RL-driven health interventions. Define MDP, agents, and experiments in YAML — no code changes needed.

## What this is

A HeartSteps-style simulator that runs configurable episodes with binary interventions (send/don't send a motivational message) and compares agent performance. Currently implements the MVP (Issue #101): Thompson Sampling, epsilon-greedy, and random agents on a 90-day × 5 steps/day episode.

## Quickstart

```bash
uv sync --dev

# Run with default config (Thompson Sampling, 450 steps)
uv run rl-health-interventions --config config/mvp.yaml --output results.csv

# Run with epsilon-greedy baseline
uv run rl-health-interventions --config config/mvp.yaml --agent epsilon_greedy --output results.csv

# Run with random baseline
uv run rl-health-interventions --config config/mvp.yaml --agent random --output results.csv
```

## How it works

The MDP is defined entirely in YAML:

```yaml
# config/mvp.yaml (simplified)
activity_levels: [sedentary, active]
actions: [send, don_t_send]
steps_per_day: 5
episode_days: 90
transition:
  sedentary:
    send: {active: 0.3, sedentary: 0.7}
    don_t_send: {active: 0.1, sedentary: 0.9}
  active:
    send: {active: 0.5, sedentary: 0.5}
    don_t_send: {active: 0.6, sedentary: 0.4}
masks:
  night: {sedentary: 1.0, active: 1.0}  # no transitions at night
```

Change the YAML to change the experiment. No code changes needed.

## Agents

| Agent | Type | Learning | CLI flag |
|-------|------|----------|----------|
| Thompson Sampling | Beta-Bernoulli bandit | Yes (posterior update) | `--agent thompson_sampling` |
| Epsilon-Greedy | Q-learning bandit | Yes (incremental average) | `--agent epsilon_greedy` |
| Random | Uniform random | No | `--agent random` |

All agents are contextual bandits in the MVP — state is accepted but not used in action selection. State-aware agents are planned for Phase 2.

## MDP formulation

See `docs/mvp/mvp_specification.tex` for the full MDP specification with transition probabilities, reward function, and agent formulations.

**Initial results** (config/mvp.yaml, 10 seeds, 450 timesteps each):

| Agent | Total Reward | Mean Reward/Step |
|-------|-------------|-----------------|
| Thompson Sampling | 161.9 ± 15.4 | 0.360 ± 0.034 |
| Epsilon-Greedy (ε=0.1) | 155.9 ± 17.7 | 0.346 ± 0.039 |
| Random | 133.6 ± 14.8 | 0.297 ± 0.033 |

## Project structure

```
src/rl_health_interventions/
  config/        # MDPConfig, YAML loader
  agents/        # Thompson Sampling, epsilon-greedy, random
  transitions/   # RuleBasedTransition (config-driven matrix)
  rewards/       # CompoundReward (config-driven)
  state.py       # StateView dataclass
  environment.py # step/reset simulation loop
  experiment.py  # run_episode + CSV output
config/          # MVP YAML config
docs/mvp/        # design.tex (MDP formulation + results)
tests/           # 96 tests (unit + integration)
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

Per `docs/ROADMAP.md`, these are deferred until after the MVP ships:

- Multi-timescale reward (immediate + delayed body measure)
- 4 user archetypes (goal-driven, social, resistant, stable)
- Burden accumulation / decay model
- Evaluation framework (bootstrap CIs, power analysis)
- Multi-feature synthetic data matched to population statistics
- Safety / ethics review

## References

- HeartSteps V2 (Klasnja et al., 2022)
- Issue #101 (MVP spec)
- `docs/ROADMAP.md` (backlog, rough guidance)
