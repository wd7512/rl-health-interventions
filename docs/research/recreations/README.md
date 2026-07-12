# Research Recreations

This directory holds recreation reports for published papers relevant to the `rl-health-interventions` framework. Each report documents headline numbers from the original paper, what we can and cannot recreate given available data and code, and concrete next actions for the project.

## Reports

| Paper | Report | Status |
|---|---|---|
| HeartSteps V1 — Klasnja et al. (2015, UbiComp; 2019, *Annals of Behavioral Medicine*) | `heartsteps-v1.md` | Complete |
| HeartSteps V2 — Liao et al. (2019, *UbiComp/ISWC*) | PR #85 (`paper_reproduction/`) | Blocked by review |
| AHRQ Population Model | `recreate-ahrq-population/` (branch) | Planned |
| Health Gym | `recreate-healthgym/` (branch) | Planned |
| MyHeartCounts | `recreate-myheartcounts/` (branch) | Planned |
| StepCountJITAI | `recreate-stepcountjitai/` (branch) | Planned |
| TS Bounds | `recreate-ts-bounds/` (branch) | Planned |

## Format

Each recreation report follows the same structure:

1. **Paper overview** — what the paper did and why it matters
2. **Headline numbers table** — key quantitative results with section citations
3. **Recreation methodology** — what can and cannot be recreated
4. **Four-question loop** — theoretical limit, linkages, improvement, action
5. **Validation assessment** — honest appraisal of what is validated
6. **Open questions** — unresolved issues for the project
7. **Citation block** — full BibTeX references
8. **Next actions** — priority-ordered, specific, testable tasks

## Related

- Statistical Analysis Plan: `docs/research/archive/statistical-analysis-plan.md`
- PR #85 reproduction (HeartSteps V2): branch `pr-85`, `paper_reproduction/`
- Framework design: `docs/design/initial_design.tex`

---

*Last updated: 2026-06-15*
