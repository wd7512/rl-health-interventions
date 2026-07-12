# Agent Instructions

This is a generic Python project template using `uv`, `ruff`, `ty`, and `pytest`.

## Commands

- Install dependencies: `uv sync --dev`
- Format: `uv run ruff format`
- Format check (CI): `uv run ruff format --check .`
- Lint: `uv run ruff check`
- Type check: `uv run ty check --exclude tests/`
- Test: `uv run pytest`
- Build: `uv build`

## Rules

- Use `uv` for dependency management and command execution.
- Keep changes small and focused.
- Add or update tests for behavior changes.
- Run `uv run ruff format --check`, `uv run ruff check`, `uv run ty check`, and `uv run pytest` before finishing code changes.
- Do not add runtime dependencies unless they are required by package behavior.
- Prefer clear, boring Python over clever abstractions.
- Use `logging` (stdlib) for all output — never `print()`. Configure via `logging.basicConfig()` in entry points.

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

## Optional Agent Support

- `opencode.json` configures project-local OpenCode behavior.
- `.opencode/agents/` contains reusable OpenCode agents.
- `.opencode/skills/` contains reusable workflow skills for AI-assisted development.
- `.opencode/plugins/write-size-guard.ts` prevents oversized generated writes.
- `.opencode/plugins-available/` contains optional telemetry plugins that can be copied into `.opencode/plugins/` when desired.
