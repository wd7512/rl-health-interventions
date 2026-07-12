# rl-health-interventions

Config-driven RL simulation framework for testing Just-In-Time Adaptive Interventions
(JITAIs) on wearable physical activity data. Aimed at real-world health intervention trials.
Python 3.11 · uv · Pydantic v2 · Polars · NumPy · Ruff · pytest · ty

## Commands

`uv sync --dev` · `uv run pytest` · `uv run ruff format .` · `uv run ruff check` · `uv run ty check --exclude tests/` · `uv run build`

## Architecture

- `agents/` RL agents — registry plugin system, subclass `Agent`, implement `select_action()`, call `register()`
- `transitions/` state transition models (rule-based, random, LLM-bootstrapped)
- `rewards/` config-driven reward formulas via safe AST parser (no `eval()`)
- `simulation/` episode runner — 90 days × 5 steps/day, `StateView` is immutable
- `data/` Polars readers + feature pipeline for 7+ wearable datasets
- `config/` Pydantic v2 schemas + YAML loaders
- `llm_bootstrapping/` generates transition probability tables via LLM (DeepSeek/OpenRouter)
- `evaluation/` metrics, shared runner, result aggregation

## Testing

pytest + `pytest-timeout` (10s default) + `pytest-xdist`. Markers: `unit`, `integration`, `regression`.

## Boundaries

**Never:** `.env` / tokens / credentials / `data/` / `dist/` / `.worktrees/`
**Ask first:** Pydantic schemas (breaks YAML) · agent/transitions/rewards registry pattern · new deps

## Docs

`docs/overview/` · `docs/decisions/` · `docs/research/` · `docs/experiments/` · `docs/sources/` · `docs/guides/` · `docs/archive/`
Source of truth: `docs/decisions/resolved-decisions-sprint-1.md` · `docs/decisions/decision-catalogue.md`
