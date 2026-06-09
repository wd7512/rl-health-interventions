# rl-health-interventions

A configurable simulation framework for testing RL-driven health interventions on wearable device data.

## Quickstart

```bash
uv sync --dev
uv run rl-health-interventions
```

## Development Commands

```bash
uv run ruff format
uv run ruff check
uv run ty check
uv run pytest
```

## Project Structure

```
src/rl_health_interventions/   # package source
docs/                          # documentation and plans
tests/                         # test suite
```

## Overview

Config-first design. Researchers define their dataset schema, MDP, agents, and experiments via config files — no source code changes needed.

Phase 1 delivers: config-driven data layer, MDP environment, rule-based user simulation, RL agent library, and experiment runner. Phase 2 (stretch) adds LLM-based user simulation.

See `docs/01 Codebase Plan.md` for full architecture.
