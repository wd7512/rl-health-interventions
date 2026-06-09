# Agent Instructions

Config-driven simulation framework for testing RL-driven health interventions on wearable device data.

## Commands

- Install dependencies: `uv sync --dev`
- Format: `uv run ruff format`
- Lint: `uv run ruff check`
- Type check: `uv run ty check`
- Test: `uv run pytest`
- Build: `uv build`

## Rules

- Use `uv` for dependency management and command execution.
- Keep changes small and focused.
- Add or update tests for behavior changes.
- Run `uv run ruff check`, `uv run ty check`, and `uv run pytest` before finishing code changes.
- Do not add runtime dependencies unless they are required by package behavior.
- Prefer clear, boring Python over clever abstractions.
- Use `logging` (stdlib) for all output — never `print()`. Configure via `logging.basicConfig()` in entry points.
- Config is JSON (Python stdlib). Do not add PyYAML unless explicitly requested.
- Every PR must have an associated issue. No orphan PRs.
- Squash merge only. Never merge commits or rebase.

## Contributing

See `CONTRIBUTING.md` for workflow details.

## Project Context

- **Stakeholders:** William Dennis (builder), Mengyan Zhang (Bristol data), Swapnil Mishra (NUS/MLGH)
- **MLGH group:** Bayesian epidemiologists. Primary: R, Stan, NumPyro, Jupyter. Python for DL (VAEs, GPs). No existing RL/LLM code.
- **Framework spec:** See spec document in Obsidian vault.
