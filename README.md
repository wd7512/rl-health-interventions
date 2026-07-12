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
  config/        # Pydantic schemas + YAML loaders
  agents/        # Thompson Sampling, epsilon-greedy, UCB, random + HeartSteps
  transitions/   # RuleBasedTransition, BootstrapTransition (LLM-estimated)
  rewards/       # CompoundReward (precomputed per-step reward)
  data/          # Synthetic data generation, Polars readers
  simulation/    # Rule-based user response model
  evaluation/    # Metrics, shared runner, result aggregation
  llm_bootstrapping/  # LLM transition table generation
  environment.py # step/reset simulation loop
  experiment.py  # run_episode + run_experiment
config/          # YAML config files
docs/
  overview/      # ROADMAP, milestones, success metrics
  decisions/     # Source of truth for MDP design decisions
  research/      # Evidence reviews, paper recreations
  experiments/   # MVP, sprint1_bootstrap, sprint1_random + results + figures
  sources/       # Dataset documentation (14 sources)
  guides/        # Planning documents
  archive/       # Stale/superseded docs
tests/           # Unit, integration, regression tests
```

## Development

```bash
uv run ruff format --check   # Format check
uv run ruff check            # Lint
uv run ty check --exclude tests/  # Type check
uv run pytest                # Tests
uv build                     # Build
```

## References

- HeartSteps V2 (Klasnja et al., 2022)
- `docs/decisions/initial_design.tex` (MDP formalisation)
- `docs/overview/ROADMAP.md` (backlog, rough guidance)
- `docs/decisions/resolved-decisions-sprint-1.md` (Sprint 1 MDP spec)
