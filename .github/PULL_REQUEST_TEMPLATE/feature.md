## Description

<!-- Briefly describe what this PR does and why. Link the issue. -->

Closes #

## Scope

- [ ] Changes limited to `docs/`, `src/`, and `tests/` only
- [ ] No config files, CI workflows, or unrelated files modified
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
- [ ] Ruff clean (`uv run ruff check .`)

## Experiments

- [ ] Experiment config YAML created in `config/experiments/`
- [ ] Experiment has been run end-to-end
- [ ] Results reviewed and make sense (not just "it ran")
- [ ] Key findings reflected in docs

## Documentation

- [ ] `initial_experiments.tex` updated with implementation details
- [ ] Compiled PDF generated and committed
- [ ] Config schema documented (fields, defaults, modes)
- [ ] Doc quality: written for someone reading for the first time — clear motivation, no unexplained jargon

## PR Hygiene

- [ ] Description lists what changed (bullet points)
- [ ] No unrelated changes bundled in
- [ ] Branch is up to date with base

## Review

- [ ] CI passes (tests, lint)
- [ ] All review threads resolved
- [ ] At least one approval before merge
