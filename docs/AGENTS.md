---
title: "AI Agent Navigation"
status: "active"
last_reviewed: "2026-07-12"
---
# AI Agent Navigation

This document helps AI coding agents navigate the project's documentation.

## Directory map

- `docs/overview/` — ROADMAP, milestones, success metrics
- `docs/decisions/` — Source of truth for MDP design decisions
- `docs/research/` — Research evidence, paper recreations, lit reviews
- `docs/experiments/` — Experiment code, configs, results, figures
- `docs/sources/` — Dataset feasibility and availability docs
- `docs/guides/` — Planning documents
- `docs/archive/` — Stale/superseded documents

## Key files

- `docs/decisions/resolved-decisions-sprint-1.md` — Sprint 1 MDP specification (source of truth)
- `docs/decisions/decision-catalogue.md` — All MDP design decisions with resolution status
- `docs/decisions/initial_design.tex` — Academic design document (long-term vision, not current spec)
- `docs/sources/data_availability_schema.md` — Master dataset reference

## Conventions

- Active docs live in topic directories; stale docs are git mv'd to `docs/archive/`.
- All active docs use YAML frontmatter with `status` and `last_reviewed` fields.
- Experiment scripts in `docs/experiments/` use relative imports from `_shared.py`.
- Dataset docs in `docs/sources/` are self-contained with minimal cross-references.
- Decision docs in `docs/decisions/` are the authoritative MDP specification.

## Cross-reference rules

- Internal links: use relative paths within `docs/`
- External references: use `docs/` relative paths from project root
- After moving files, update all cross-references in both directions
