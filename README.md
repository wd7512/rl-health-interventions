# rl-health-interventions

A config-driven RL simulation framework for testing health interventions. Define MDPs, agents, and experiments in YAML — no source code changes needed.

## Quickstart

```bash
uv sync --dev

# Run with default config
uv run rl-health-interventions --config config/rule_based.yaml --output results.csv

# Override agent from command line
uv run rl-health-interventions --config config/rule_based.yaml --agent epsilon_greedy --output results.csv
```

## How it works

The MDP (states, actions, transitions, reward) and experiment parameters (agents, episodes, seeds) are defined entirely in YAML. Change the config to change the experiment — no code changes needed.

See `config/` for example configs, and `docs/` for the MDP formalisation.

## Project structure

```
src/rl_health_interventions/
  config/        # MDPConfig, AgentConfig, YAML loader + validators
  agents/        # Thompson Sampling, epsilon-greedy, UCB, random + contextual variants
  transitions/   # RuleBasedTransition (config-driven matrix)
  rewards/       # CompoundReward (precomputed per-step reward)
  data/          # Synthetic data generation, dataset loaders
  simulation/    # Rule-based user response model
  state.py       # StateView dataclass
  environment.py # step/reset simulation loop
  experiment.py  # run_episode + run_experiment
  logging.py     # Stdlib logging setup
config/          # YAML config files
docs/experiments/mvp/          # MVP MDP formulation + initial results
docs/experiments/              # Extensions: state space, action space, reward function
docs/decisions/                # Long-term design vision
docs/sources/                  # Dataset documentation (14 sources)
docs/overview/                 # Roadmap, sub-phase plans
tests/           # Unit, integration, regression tests
```

## Development

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
uv build
```

## References

- HeartSteps V2 (Klasnja et al., 2022)
- `docs/decisions/initial_design.tex` (MDP formalisation)
- `docs/overview/ROADMAP.md` (backlog, rough guidance)
