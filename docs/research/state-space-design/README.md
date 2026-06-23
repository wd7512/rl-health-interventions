---
title: "State Space Design — Research Artifacts"
status: "draft v0.1"
date: "2026-06-23"
purpose: "Collate evidence and reference configurations informing state space design"
related: "lit-review-state-action-space.md · action-space-design/ · docs/initial_design.tex §3"
---

# State Space Design — Research Artifacts

> Reference configurations and evidence reviews that inform the state space
> for the rl-health-interventions project. Each file is descriptive — it
> reports what the literature has used, not what we will use.

## Files

| File | Content |
|------|---------|
| `reference-configs.md` | 5 literature-derived system configurations: state features and cardinality for each |
| `step-bin-evidence.md` | Dose-response evidence for step count thresholds (4 bins vs binary vs continuous) |
| `hidden-state-evidence.md` | Literature on mood/stress/sleep as latent state variables in activity MDPs |

## Open decisions (for state space)

These are questions the literature does not settle. No recommendation is made
here — each is flagged for resolution on the implementation branch.

1. **Hidden mood state**: Evidence for mood/stress as a latent state is thin.
   No published RL system models it. See `hidden-state-evidence.md`.

2. **Factored vs flat representation**: Factored (contextual bandit features)
   is preferred but the codebase currently uses flat state keys in
   `per_step_reward`. Migration path is unclear.

3. **Trend computation**: The `trend` dimension (increasing/stable/decreasing)
   requires a rolling window (3–7 days). Window size and computation method
   are not settled.

4. **Step bin boundaries**: 4 thresholds (<4k, 4–7k, 7–10k, ≥10k) are
   supported by evidence for *mortality* outcomes. Optimality for *activity
   engagement* outcomes may differ.

## Cross-references

- `lit-review-state-action-space.md` — original synthesis
- `action-space-design/` — companion research on action representation
- `docs/initial_design.tex §3` — MDP design containing state specifications
