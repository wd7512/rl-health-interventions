---
title: "Paper Recreation Reports"
status: "active"
date: "2026-06-14"
purpose: "One recreation report per paper in the literature review; each applies the four-question loop"
---

# Paper Recreation Reports

This directory holds recreation reports for each paper that the
project cites or could cite. Each report:

1. States the paper's headline result with exact numbers
2. Applies the four-question loop:
   - Is this at the theoretical limit?
   - How does it link to other papers and this work?
   - Is there still room for improvement?
   - Take action (concrete next step)
3. Is honest about what the recreation validates and what it does
   not (e.g. we cannot "run" a theorem)

The recreations are **research artefacts**, not implementations.
They inform PRs and the paper's methodology section.

## Reports in this directory

| Paper | Status | PR/branch | Headline result |
|---|---|---|---|
| Klasnja et al. 2019 (HeartSteps V1) | pending (opencode) | research/recreate-heartsteps-v1 | ATE +13.9 steps/30min, NS, 72% availability |
| Russo & Van Roy 2018 (TS bounds) | draft v0.1 | research/recreate-ts-bounds | `O(√(dT log T))` Bayesian regret for TS |
| (more forthcoming) | | | |

## Conventions

- No code, no `src/` changes
- No edits to `initial_design.tex` (gated)
- Use the same citation block format as in `docs/references.bib`
- Each report's "Take action" section produces 3-5 concrete,
  testable actions the project should take
- Reports are honest: if a number cannot be verified, say so

## Connection to other research docs

- The **statistical analysis plan** (PR #86) sets the conventions
  for how the project will report recreation results
- The **simulator validation decision tree** (PR #87) determines
  what "recreation" means for synthetic-data papers vs.
  real-data papers
- The **online/offline framing** (PR #88) determines whether a
  recreation is a numerical reproduction (online framing) or a
  validation exercise (offline framing)
- The **PR #85 review** (PR #90) is the first recreation report
  and sets the template for this directory
