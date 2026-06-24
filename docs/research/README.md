---
title: "Research Artifacts"
status: "active"
date: "2026-06-14"
purpose: "Pre-implementation research outputs for the rl-health-interventions project"
---

# Research Artifacts

This directory holds research-process documents that support the
project but are not implementation artefacts. They cover decision
records, paper recreations, and historical research batches — all
written in Markdown, versioned with the repo, and reviewed via PR.

## Index

| Path | Type | Description | Status |
|------|------|-------------|--------|
| [decision-trees/online-vs-offline-rl.md](decision-trees/online-vs-offline-rl.md) | Decision record | Resolves online vs offline RL framing — recommends simulation-based policy evaluation | Closed |
| [recreations/](recreations/) | Paper recreations | Cross-paper synthesis and individual recreation reports for 8 published papers | Active |
| [recreations/SYNTHESIS.md](recreations/SYNTHESIS.md) | Synthesis | Aggregated findings from 8 paper recreations | Active |
| [archive/](archive/) | Archived | Historical research artifacts from the June-14 batch, retained for reference | Archived |

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
- Research documents are *inputs* to design decisions and
  implementation, not the decisions themselves.

## Review process

- Internal review: open a PR, assign yourself, request review from
  Mengyan (clinical / outcomes) or Swapnil (MDP / RL methodology) as
  appropriate.
- External review: invite one external RL-for-health researcher to
  review the analysis plan before locking v1.0. The audit flagged
  that no external review process currently exists; this is the seed
  of one.

## Related directories

- [.reviews_and_reports/](../../.reviews_and_reports/) — code and PR review artifacts
- [docs/sources/](../sources/) — dataset feasibility and reference data
- [docs/design/](../design/) — academic design documents
- [docs/experimental_phases/](../experimental_phases/) — experiment specifications and results

## Why a separate directory

`docs/` already contains:

- `code/` — implementation design
- `sources/` — dataset feasibility and references
- `ROADMAP.md` — implementation milestones
- `initial_design.tex` — the academic design document (gated)

`docs/research/` is for *research-process* artefacts that don't fit
any of those. The split keeps the gate on `initial_design.tex`
clean and gives research docs a clear, single home.
