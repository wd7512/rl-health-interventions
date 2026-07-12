---
title: "Archive Log"
status: "active"
last_reviewed: "2026-07-12"
---
# Archive Log

This document tracks documents that have been superseded or archived.

## Why this exists

The docs/ directory was restructured from a phase-based layout (`plans/`, `research/`, `sprint1/`, `sources/`, `experimental_phases/`) to a topic-based layout (`overview/`, `research/`, `decisions/`, `experiments/`, `sources/`, `guides/`). Documents that were stale, superseded, or not aligned with the current Sprint 1 MDP specification (3 step bins, 4 actions) were moved here.

## Archived files

### plans/ — Superseded implementation blueprints
| Original path | Archive path | Reason |
|---|---|---|
| `docs/plans/stale/code_design.md` | `docs/archive/plans/code_design.md` | Early architecture blueprint, superseded by implementation |
| `docs/plans/stale/codebase_plan.md` | `docs/archive/plans/codebase_plan.md` | Overall plan, superseded by actual implementation |
| `docs/plans/stale/issue-101-mvp.md` | `docs/archive/plans/issue-101-mvp.md` | MVP plan, superseded by Issue #101 implementation |
| `docs/plans/stale/phase_1_execution_plan.md` | `docs/archive/plans/phase_1_execution_plan.md` | Early phase plan, superseded |
| `docs/plans/stale/subphase_1b_mdp_environment.md` | `docs/archive/plans/subphase_1b_mdp_environment.md` | MDP environment plan, superseded |
| `docs/plans/stale/subphase_1d_agent_library.md` | `docs/archive/plans/subphase_1d_agent_library.md` | Agent library plan, superseded |
| `docs/plans/stale/subphase_1e_experiment_runner.md` | `docs/archive/plans/subphase_1e_experiment_runner.md` | Experiment runner plan, superseded |
| `docs/plans/stale/learned_transitions.md` | `docs/archive/plans/learned_transitions.md` | Learned transitions plan, deferred to Phase 2 |
| `docs/plans/stale/2026-06-14_research-assistant-plan.md` | `docs/archive/plans/2026-06-14_research-assistant-plan.md` | Research PR plan, completed |

### research/ — Archived research artifacts
| Original path | Archive path | Reason |
|---|---|---|
| `docs/research/archive/framework-comparison.md` | `docs/archive/research/framework-comparison.md` | Positioning table, archived |
| `docs/research/archive/simulator-validation.md` | `docs/archive/research/simulator-validation.md` | Simulator validation strategy, archived |
| `docs/research/archive/statistical-analysis-plan.md` | `docs/archive/research/statistical-analysis-plan.md` | Statistical analysis plan, archived |
| `docs/research/archive/SUPERVISOR_SUMMARY.md` | `docs/archive/research/SUPERVISOR_SUMMARY.md` | Supervisor summary, archived |

### Other archived docs
| Original path | Archive path | Reason |
|---|---|---|
| `docs/contents.md` | `docs/archive/contents.md` | Replaced by docs/README.md navigation hub |
| `docs/sprint1/funding-proposal.md` | `docs/archive/funding-proposal.md` | Draft funding proposal, not current |
| `ROADMAP.md` (root) | `docs/overview/project-roadmap.md` | Moved to topic-based structure |

## Files never tracked by git

The following files existed in `docs/experimental_phases/ideas/` but were never committed to git:
- `docs/experimental_phases/ideas/README.md`
- `docs/experimental_phases/ideas/experimental_template/` (template stubs)
- `docs/experimental_phases/ideas/learned/` (placeholder spec)
- `docs/experimental_phases/ideas/llm/` (placeholder spec)

These remain untracked in the working tree and were not part of the restructuring.

## Missing files noted during restructuring

The following files are referenced in active docs but do not exist:
- `docs/plans/subphase_1a_data_layer.md` — Referenced in `docs/guides/plans-overview.md` (old `plans/README.md`) as active, but never written
- `docs/plans/subphase_1c_user_simulation.md` — Referenced in `docs/guides/plans-overview.md` and `docs/overview/ROADMAP.md` as active, but never written

These appear in `docs/archive/plans/` as `stale/` variants only for subphase_1b, 1d, and 1e. Subphases 1a and 1c were never documented.
