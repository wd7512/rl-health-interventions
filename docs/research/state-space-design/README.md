---
title: "State Space Design — Research Artifacts"
status: "active"
date: "2026-06-23"
purpose: "Collate evidence and reference configurations informing state space design"
related: "decision-catalogue.md · action-space-design/ · docs/initial_design.tex §3"
---

# State Space Design — Research Artifacts

> Reference configurations and evidence reviews that inform the state space
> for the rl-health-interventions project. Each file is descriptive — it
> reports what the literature has used, not what we will use.

## Files

| File | Content |
|------|---------|
| `reference-configs.md` | 6 literature-derived system configurations: state features and cardinality for each |
| `step-bin-evidence.md` | Dose-response evidence for step count thresholds (4 bins vs binary vs continuous) |
| `hidden-state-evidence.md` | Literature on mood/stress/sleep as latent state variables in activity MDPs |

## Cross-references

- [decision-catalogue.md](../decision-catalogue.md) — all open design decisions
- [action-space-design/](../action-space-design/) — companion research on action representation
- [docs/initial_design.tex §3](../../initial_design.tex) — MDP design containing state specifications
- [decision-trees/online-vs-offline-rl.md](../decision-trees/online-vs-offline-rl.md) — upstream framing decision
