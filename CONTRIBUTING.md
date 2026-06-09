# Contributing to rl-health-interventions

## Branch Model

**Feature-branch model.** `main` is the only persistent branch.

- Feature branches: `feat/<description>`, `fix/<description>`, `docs/<description>`
- Branch from `main`, merge to `main`, delete after merge
- No `dev` branch — all integration goes through `main`

## Workflow

1. **Create an issue** describing what you want to do
2. **Create a branch** from `main`: `git checkout -b feat/my-feature`
3. **Make changes** with focused commits
4. **Open a PR** targeting `main`. Reference the issue in the PR body: `Closes #N`
5. **CI must pass** (ruff, ty, pytest) — all checks must be green before merge
6. **Review required** — conversation threads must be resolved
7. **Squash merge** — PRs are squash-merged, not merged with merge commits
8. **Branch auto-deleted** after merge (repo setting enabled)

## Rules

- **Every PR must have an associated issue.** The issue is the spec/requirement; the PR is the implementation. No orphan PRs.
- **Never commit directly to `main`.** All changes go through PRs.
- **Run the full check suite before pushing:**
  ```bash
  uv run ruff format
  uv run ruff check
  uv run ty check
  uv run pytest
  ```
- **Keep changes small and focused.** One feature or fix per PR.
- **Commit messages:** Use conventional format: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`

## Development Setup

```bash
uv sync --dev
uv run pytest
```

## Code Style

- Formatting: `ruff format` (line length 88)
- Linting: `ruff check`
- Type checking: `ty`
- Testing: `pytest`
- No runtime dependencies unless absolutely required
