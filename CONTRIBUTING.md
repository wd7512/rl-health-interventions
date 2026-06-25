# Contributing to rl-health-interventions

## Workflow

1. **Create an issue** using the Task template before starting work.
2. **Flesh out the issue** — What / Why / Completion Criteria / Evidence. If it's just an idea, label it `needs-refinement` and flesh it out later before implementation.
3. **Open a PR** that links the issue with `Closes #N`.
4. **Ensure CI passes** — ruff, ty, pytest.
5. **Resolve all review comments** before merge.

## Rules

- Every PR **must** link an issue that meets the task template.
- Issues labeled `needs-refinement` cannot be linked to PRs until fleshed out.
- No direct commits to `main` — all changes go through PRs.
- Run `uv run ruff format`, `uv run ruff check`, `uv run ty check`, and `uv run pytest` before pushing.

## Issue Template

All issues must answer:

- **What do you want to do?** (one sentence: "I want to X")
- **Why?** (how does this benefit the research?)
- **Completion criteria** (how do we know this is done?)
- **Evidence** (papers, notes, data — optional but encouraged)

Raw ideas get the `needs-refinement` label until they meet this template.

## Labels

| Label | Meaning |
|-------|---------|
| `state-space` | State space design and implementation |
| `action-space` | Action space design and implementation |
| `reward` | Reward function design and components |
| `agents` | RL agent implementations |
| `data-pipeline` | Data loading, pipeline, integration, and evaluation |
| `llm` | LLM-based components and high-dimensional spaces |
| `write-up` | Documentation, reports, and paper drafts |
| `backlog` | Not in current sprint — pick up when blocked |
| `needs-refinement` | Idea not yet fleshed out — does not meet template |
| `bug` | Something isn't working |
| `documentation` | Improvements or additions to documentation |

## Code Style

- Python 3.11+
- `uv` for dependency management
- `ruff` for formatting and linting
- `ty` for type checking
- `logging` (stdlib) for all output — never `print()`
- Tests in `tests/unit/` and `tests/integration/`
