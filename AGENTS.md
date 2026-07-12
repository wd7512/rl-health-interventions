# rl-health-interventions

Simulation framework for testing RL-driven health interventions on wearable device data.
Python 3.11 · uv · Pydantic v2 · Polars · NumPy · Ruff · pytest · ty

## Commands

```
uv sync --dev                  # Install dependencies
uv run pytest                  # All tests
uv run pytest -m unit          # Unit tests only (no network)
uv run pytest path/to.py       # Single file
uv run ruff format .           # Auto-format
uv run ruff format --check .   # Format check (CI)
uv run ruff check              # Lint
uv run ruff check --fix .      # Lint + auto-fix
uv run ty check --exclude tests/  # Type check
uv run build                   # Build distribution
```

## Architecture

`agents/` RL agents (registry plugin) · `transitions/` state models · `rewards/` reward functions
`simulation/` runner · `data/` Polars readers + feature pipeline · `config/` Pydantic + YAML
`llm_bootstrapping/` LLM transition bootstrapping · `evaluation/` metrics + result aggregation

Tests: `tests/unit/` · `tests/integration/` (may hit network/disk) · `tests/regression/`

## Code Style

Ruff line-length 88, Python 3.11+. Type hints mandatory (UP + FA). Aliases: `np`/`pl` (ICN).
Early returns preferred (RET). Max complexity 5, branches 6, statements 30 per function.
`logging` only — never `print()`. Pydantic v2 for config schemas. Named exports, no star imports.

## Testing

pytest + `pytest-timeout` (10s default) + `pytest-xdist`. Markers: `unit`, `integration`, `regression`.

## Boundaries

**Never:** `.env` / tokens / credentials / `data/` / `dist/`
**Ask first:** Pydantic schemas (breaks YAML) · registry pattern · new deps · test timeout changes

## Docs

`docs/overview/` · `docs/decisions/` · `docs/research/` · `docs/experiments/` · `docs/sources/` · `docs/guides/` · `docs/archive/`

Source of truth: `docs/decisions/resolved-decisions-sprint-1.md` · `docs/decisions/decision-catalogue.md` · `docs/sources/data_availability_schema.md`
