---
title: "Research Artifacts"
status: "active"
date: "2026-06-14"
purpose: "Pre-implementation research outputs for the rl-health-interventions project"
---

# Research Artifacts

This directory holds research-side documents that support the project
but are not implementation artefacts. They are written in markdown,
versioned with the repo, and reviewed via PR like any other change.

## What's in here

| File / Directory | Type | Status | Owner |
|---|---|---|---|---|
| `statistical-analysis-plan.md` | Pre-registration | draft v0.1 | W. Dennis |
| `lit-review-state-action-space.md` | Literature review (synthesis) | draft v0.1 | W. Dennis |
| `action-space-design/` | Action space reference configs + evidence | draft v0.1 | W. Dennis |
| `state-space-design/` | State space reference configs + evidence | draft v0.1 | W. Dennis |
| `decision-trees/` | Decision records | (forthcoming) | W. Dennis |

## Conventions

- Documents are written for a research audience, not engineering.
  Methodology, statistical reasoning, and external review-readiness
  take priority over terseness.
- Documents are *not* implementation plans. They specify *what* the
  research claims and *how* it will be evaluated, not *how to code
  it*.
- Documents reference `docs/initial_design.tex` and `docs/ROADMAP.md`
  where relevant, but never modify them.
- Documents are *versioned*. Changes are logged in the document's
  amendment log, not by rewriting history.

## Review process

- Internal review: open a PR, assign yourself, request review from
  Mengyan (clinical / outcomes) or Swapnil (MDP / RL methodology) as
  appropriate.
- External review: invite one external RL-for-health researcher to
  review the analysis plan before locking v1.0. The audit flagged
  that no external review process currently exists; this is the seed
  of one.

## Why a separate directory

`docs/` already contains:

- `code/` — implementation design
- `sources/` — dataset feasibility and references
- `ROADMAP.md` — implementation milestones
- `initial_design.tex` — the academic design document (gated)

`docs/research/` is for *research-process* artefacts that don't fit
any of those. The split keeps the gate on `initial_design.tex`
clean and gives research docs a clear, single home.
