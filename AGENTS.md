# rl-health-interventions

Simulation framework for testing reinforcement-learning-driven health interventions
on wearable device data. Python 3.11, uv, Pydantic v2, Polars, NumPy.

## Commands

- `uv sync --dev`: Install all dependencies
- `uv run pytest`: Run full test suite
- `uv run pytest tests/unit/`: Unit tests only (fast, no I/O)
- `uv run pytest tests/unit/agents/test_thompson_sampling.py -v`: Single test file
- `uv run pytest -m "not integration"`: Skip network/disk tests
- `uv run ruff format .`: Auto-format
- `uv run ruff format --check .`: Format check (CI)
- `uv run ruff check`: Lint
- `uv run ruff check --fix .`: Lint and auto-fix
- `uv run ty check --exclude tests/`: Type check
- `uv run build`: Build distribution

## Architecture

- `src/rl_health_interventions/agents/`   RL agent implementations (registry-based plugin system)
- `src/rl_health_interventions/transitions/` State transition models
- `src/rl_health_interventions/rewards/`   Reward function implementations
- `src/rl_health_interventions/simulation/` Simulation runner
- `src/rl_health_interventions/data/`      Data loading, Polars readers, feature pipeline
- `src/rl_health_interventions/config/`    Pydantic config schemas + YAML loaders
- `src/rl_health_interventions/llm_bootstrapping/` LLM-assisted transition bootstrapping
- `src/rl_health_interventions/evaluation/` Metrics, shared runner, result aggregation
- `tests/unit/`                            Fast, isolated tests (no network)
- `tests/integration/`                     May hit network or disk
- `tests/regression/`                      Baseline regression checks

## Code Style

- Ruff line-length 88, target Python 3.11+
- Type hints on all function signatures (mandatory via ruff UP + FA rules)
- Use `np` for NumPy, `pl` for Polars (enforced via ICN aliases)
- Prefer early returns over nested conditionals (RET rules enforced)
- Max cyclomatic complexity 5 per function; max 6 branches, 30 statements
- No `print()` — use `logging` with `logging.basicConfig()` in entry points
- Pydantic v2 for all config schemas and data validation
- Named exports from `__init__.py`, no star imports

## Testing

- pytest with `pytest-timeout` (10s default), `pytest-xdist` for parallelism
- Markers: `unit` (fast, no I/O), `integration` (may hit network/disk), `regression`
- Run a single marker: `uv run pytest -m unit`
- Tests enforce strict ruff rules; test files get relaxed ARG, PLC0415, PLR2004 ignores

## Boundaries

### Never

- Commit `.env` files, HuggingFace tokens, or Kaggle credentials
- Modify `data/` raw data directories or `dist/` build artifacts
- Touch `.worktrees/` or `.reviews_and_reports/`
- Add runtime dependencies without explicit approval — this is a lean research package
- Use `print()` in source code (logging only)

### Ask First

- Changing Pydantic config schemas in `config/` (breaks YAML configs)
- Modifying the agent/transitions/rewards registry pattern in `__init__.py` files
- Adding new dependencies to `pyproject.toml`
- Changing test timeout or marker definitions

## Docs Navigation

- `docs/overview/` — ROADMAP, milestones, success metrics
- `docs/decisions/` — Source of truth for MDP design decisions
- `docs/research/` — Research evidence, paper recreations, lit reviews
- `docs/experiments/` — Experiment code, configs, results, figures
- `docs/sources/` — Dataset feasibility and availability docs
- `docs/guides/` — Planning documents
- `docs/archive/` — Stale/superseded documents

**Key files:**
- `docs/decisions/resolved-decisions-sprint-1.md` — Sprint 1 MDP specification (source of truth)
- `docs/decisions/decision-catalogue.md` — All MDP design decisions with resolution status
- `docs/decisions/initial_design.tex` — Academic design document (long-term vision)
- `docs/sources/data_availability_schema.md` — Master dataset reference

**Conventions:**
- Active docs use YAML frontmatter with `status` and `last_reviewed` fields.
- Experiment scripts in `docs/experiments/` use relative imports from `_shared.py`.
- Decision docs in `docs/decisions/` are the authoritative MDP specification.
- Internal links: use relative paths within `docs/`.
- External references: use `docs/` relative paths from project root.
