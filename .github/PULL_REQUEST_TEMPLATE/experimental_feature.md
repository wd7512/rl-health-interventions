## Description

<!-- Briefly describe what this PR does and why. Link the issue. -->

Closes #

## Scope

- [ ] Changes limited to `docs/`, `src/`, and `tests/` only
- [ ] No project config files (`pyproject.toml`, CI workflows) modified
- [ ] PR targets the correct base branch for this work

## Issue

- [ ] PR body starts with `Closes #<issue>`
- [ ] Issue describes the problem being solved
- [ ] PR scope matches the issue — no scope creep

## Implementation

- [ ] Code compiles and runs without errors
- [ ] Backward compatibility preserved where expected
- [ ] No hardcoded stale values — defaults live in config schema
- [ ] ABC signatures updated consistently across all affected modules

## Tests

- [ ] New logic covered by unit tests (happy path + edge cases)
- [ ] Existing tests updated for any signature changes
- [ ] Full suite passes locally (`uv run pytest tests/ -q`)
- [ ] MVP regression test passes (`uv run pytest tests/integration/test_mvp_end_to_end.py -q`)
- [ ] Ruff format clean (`uv run ruff format --check .`)
- [ ] Ruff lint clean (`uv run ruff check .`)
- [ ] Type check clean (`uv run ty check --exclude tests/`)

## Experiments

- [ ] Experiment folder created in `docs/<feature>/` following MVP structure
- [ ] `.tex` documentation written with motivation, method, and results
- [ ] Compiled `.pdf` committed
- [ ] Runner script (`.py`) included alongside config
- [ ] Experiment has been run end-to-end
- [ ] Results reviewed and make sense (not just "it ran")
- [ ] Key findings reflected in docs

## Documentation

- [ ] Config schema documented (fields, defaults, modes)
- [ ] Doc quality: written for someone reading for the first time — clear motivation, no unexplained jargon

## PR Hygiene

- [ ] Description lists what changed (bullet points)
- [ ] No unrelated changes bundled in
- [ ] Branch is up to date with base

## Review

- [ ] CI passes (tests, lint)
- [ ] All review threads resolved
