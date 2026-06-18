---
name: dev-workflow
description: Use when making a non-trivial code, test, configuration, or documentation change that should be planned, implemented, and verified end-to-end.
---

# Dev Workflow

Use this skill for disciplined Python project changes.

## Workflow

1. Scope the goal and inspect relevant files before editing.
2. Create a worktree under repos/.worktrees/ for the change.
3. Create or update tests for behavior changes via delegation to the @python-engineer subagent.
4. Implement the smallest correct change via delegation to the @python-engineer subagent.
5. Run formatting, linting, tests, and build checks.
6. Commit the change with a clear message and push to the feature branch.
7. Clear the worktree and delete any temporary files.
8. Summarize what changed, why, and how it was verified.

## Standard Checks

```bash
uv run ruff format --check
uv run ruff check --fix
uv run pytest
uv build
```

## Rules

- Use `uv` for all Python commands.
- Keep commits focused.
- Do not add dependencies without a clear reason.
- If evidence contradicts the plan, stop and revise the plan.
