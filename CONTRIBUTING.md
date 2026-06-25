# Contributing to rl-health-interventions

## Workflow

1. **Create an issue** using the Task template (YAML form) before starting work.
2. **Flesh out the issue** — check prerequisites, write the exact-format sentence, define completion criteria.
3. **Remove `needs-refinement` label** when the issue meets the template.
4. **Open a PR** that links the issue with `Closes #N`.
5. **Ensure CI passes** — ruff, ty, pytest.
6. **Resolve all review comments** before merge.

## Rules

- Every PR **must** link an issue that meets the task template.
- Issues labeled `needs-refinement` cannot be linked to PRs until fleshed out.
- No direct commits to `main` — all changes go through PRs.
- Run `uv run ruff format`, `uv run ruff check`, `uv run ty check`, and `uv run pytest` before pushing.
- `blank_issues_enabled: false` — all issues must use the template.

## Issue Template

All issues must follow the YAML form structure:

1. **Prerequisites** (checkboxes, required):
   - ☐ I have searched existing issues and found no duplicate
   - ☐ I have provided a clear one-sentence summary

2. **One-sentence summary** (required, exact format):
   > I want to [action] to achieve [goal] to benefit the research in [way].

3. **Context & motivation** (required, min 20 chars):
   Why does this matter? Link papers, notes, prior discussions.

4. **Completion criteria** (required, checklist):
   - [ ] Specific condition 1
   - [ ] ruff/ty clean, tests pass

5. **Evidence** (optional):
   Papers, notes, data, links — supports the decision.

6. **Dependencies & notes** (optional):
   Blockers, gotchas, suggested approach.

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
| `needs-triage` | Newly opened, awaiting triage |
| `bug` | Something isn't working |
| `documentation` | Improvements or additions to documentation |

## Code Style

- Python 3.11+
- `uv` for dependency management
- `ruff` for formatting and linting
- `ty` for type checking
- `logging` (stdlib) for all output — never `print()`
- Tests in `tests/unit/` and `tests/integration/`
