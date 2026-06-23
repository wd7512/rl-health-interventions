---
title: "Action Space Design — Research Artifacts"
status: "draft v0.1"
date: "2026-06-23"
purpose: "Collate evidence and reference configurations informing action space design"
related: "lit-review-state-action-space.md · state-space-design/ · docs/initial_design.tex §3"
---

# Action Space Design — Research Artifacts

> Reference configurations and evidence reviews that inform the action space
> for the rl-health-interventions project. Each file is descriptive, not
> prescriptive — it reports what the literature has done, not what we will do.
> Design decisions are left to the implementation branches.

## Files

| File | Content |
|------|---------|
| `reference-configs.md` | 5 literature-derived system configurations: states, actions, reward, transition type for each |
| `non-activity-interventions.md` | Evidence for journaling, sleep hygiene, and social encouragement as intervention actions |
| `action-burden-evidence.md` | Existing burden/cost models from simulators and trials |

## Open decisions (for action space)

These are questions the literature does not settle. No recommendation is made
here — each is flagged for resolution on the implementation branch.

1. **Non-activity action reward**: Journaling and sleep hygiene do not increase
   step count directly. Under a step-only reward, a learning agent would avoid
   them. Three approaches exist in the literature:
   - Placeholder mood/sleep signal in the reward
   - Defer to Phase 2
   - Include for post-hoc evaluation only
   See `non-activity-interventions.md` for causal evidence that informs this
   choice.

2. **Burden penalty magnitudes**: The burden values assigned to each action
   (0.0–0.25) are drawn from general design guidance (Trella 2022), not from
   specific empirical estimates. See `action-burden-evidence.md` for what
   existing systems have used.

3. **Fatigue model**: Whether fatigue is a separate heuristic penalty or
   collapses into a hidden mood state is unresolved. This interacts with the
   state-space design — see `state-space-design/hidden-state-evidence.md`.

## Cross-references

- `lit-review-state-action-space.md` — original synthesis that motivated these
  artifacts
- `state-space-design/` — companion research on state representation
- `docs/initial_design.tex §3` — MDP design containing action specifications
