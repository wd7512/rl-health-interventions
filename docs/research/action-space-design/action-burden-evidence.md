---
title: "Action Burden and Fatigue — Evidence Review"
status: "draft v0.1"
date: "2026-06-23"
purpose: "Review how existing simulators and trials model action burden, cost, and user fatigue"
---

# Action Burden and Fatigue — Evidence Review

> This file reviews how published systems have modelled the cost of
> delivering an action (burden penalty) and the cumulative effect of
> consecutive actions (fatigue). It is descriptive — it reports what
> others have done, not what we should do.

## 1. Burden models in existing systems

### Trella et al. (2022, Algorithms)

The most directly relevant treatment. Trella proposes:

- Each action has a **static burden penalty** (conceptually, a per-action cost)
- Cumulative burden increases with consecutive non-idle actions
- Burden decays during idle periods
- No specific decay rate or penalty magnitudes are given — these are left
  to the implementer

The paper states: "the burden associated with an intervention action should
be reflected in the reward function, either as a direct penalty or as a
constraint on action frequency." No empirical values.

### StepCountJITAI simulation (Karine 2024)

- Models burden as a linear accumulator: `B_{t+1} = B_t + b(a_t) - d * idle`
- `b(a)` ranges from 0.1–0.5 depending on action intensity
- `d = 0.2` (decay per idle step)
- Resets daily (episode boundary)
- **Linear penalty** applied to reward: `R = step_reward - B_t`

This is the closest existing implementation to our proposed model.

### HeartSteps V2 (Liao 2019)

- Does not model explicit burden
- Action cost is implicit: fewer delivered suggestions = longer system life
  before user disengagement
- No per-action penalty in the reward function
- The bandit formulation treats all actions as cost-free

## 2. Fatigue models in existing systems

No published RL-for-health system explicitly models a fatigue penalty
separate from a burden accumulator. The term "fatigue" appears in:

- **Trella 2022**: Conceptual — "users may experience intervention fatigue
  from repeated notifications" — but no formal model
- **StepCountJITAI**: Subsumed under the burden accumulator (linear increase,
  decay during idle)
- **HeartSteps**: Not modelled; the 5 decision-point/day cadence and 90-day
  horizon implicitly cap total interventions

## 3. Burden penalty values used in the literature

| Source | Context | Values used | Basis |
|--------|---------|-------------|-------|
| Trella (guidelines) | Conceptual | Not specified | None (framework) |
| StepCountJITAI | Simulation | 0.1–0.5 (per action) | Tuned to match HeartSteps suggestion rate |
| Current MVP (design) | Design | 0.0–0.3 | Plausible ranges — not grounded |
| Proposed in lit review | Design | 0.0–0.25 | Heuristic, not grounded |

**Conclusion**: Burden penalty magnitudes are universally heuristic.
No published system has empirically validated its burden values.

## 4. Summary: What we know and don't know

| Question | Known | Not known |
|----------|-------|-----------|
| Should actions have a cost? | Yes — Trella, JITAI guidelines | — |
| What functional form? | Linear accumulator used in StepCountJITAI | Whether linear is correct |
| What decay rate? | d=0.2 (StepCountJITAI) | Whether this generalises |
| When does fatigue reset? | Daily (StepCountJITAI) | Per-episode vs per-day vs per-week |
| Specific penalty values? | — | No empirical basis exists |
| Fatigue vs burden: same construct? | — | Untested in literature |

These are tradeoffs, not gaps. The literature does not settle any of them,
so implementation choices should be documented and reported as sensitivity
analyses.
