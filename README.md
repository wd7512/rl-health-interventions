# rl-health-interventions

A configurable simulation framework for testing RL-driven health interventions on wearable device data.

## What This Is

Swapnil's team plugs in their datasets, specifies the MDP and hypotheses via JSON config files, and runs experiments comparing intervention policies.

**Phase 1:** Foundational framework — config-driven data layer, MDP environment, rule-based user simulation, RL agent library, experiment runner.

**Phase 2:** LLM-based user simulation, validation against real data, LLM-augmented experiments.

## Quickstart

```bash
uv sync --dev
uv run rl-health-interventions
```

## Development

```bash
uv run ruff format
uv run ruff check
uv run ty check
uv run pytest
uv build
```

## Configuration

The framework is config-driven. Define your data schema, MDP, and experiment in JSON files. No code changes needed for new datasets or experiments.

See `CONTRIBUTING.md` for workflow and code style.

## License

MIT
