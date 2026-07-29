# PEARL Random Experiment

Testing the 4-arm PEARL experiment with random transitions (for validation).

## Overview

This experiment replicates PEARL's 4-arm RCT design:
- **Control:** FixedAgent (idle, no nudges)
- **Random:** RandomAgent (uniform action selection)
- **Fixed:** ComBWeightedFixedAgent (COM-B barrier-score weighted sampling)
- **RL:** EpsilonGreedyAgent (ε=0.3, contextual bandit)

## Configuration

- `configs/pearl_random.yaml` — Random transitions (for testing)
- `configs/pearl_bootstrap.yaml` — Bootstrap transition tables

## State Space (108 states)

5 dynamic variables:
- `recent_steps_mean`: low, moderate, high
- `recent_walk_pattern`: low, high
- `morning_steps_ratio`: morning, balanced, evening
- `day_of_week`: weekday, weekend
- `burden`: low, medium, high

## Action Space (13 actions)

1 idle + 12 COM-B actions (6 themes × 2 delivery times):
- Themes: ability, perceived_benefit, planning, prioritization, social_opportunity, physical_opportunity
- Times: morning, afternoon

## Running

```bash
# Quick test (5 seeds, ~30s)
uv run python docs/experimental_phases/pearl_random/run_experiments.py --seeds 5

# Full benchmark (50 seeds, ~5min)
uv run python docs/experimental_phases/pearl_random/run_experiments.py --seeds 50 --output docs/experimental_phases/pearl_random/results --json

# Regression test
uv run pytest regression-suite/test_pearl_random.py -v
```

## Known Limitations

- Random transitions produce degenerate results (~54% sustained burden floor regardless of agent policy)
- Bayesian P-success burden requires structured transition tables (pearl_bootstrap.yaml)
- See atomic decomposition plan for roadmap to structured transitions
