# Sim Francisco — Research Connections to rl-health-interventions

**Date:** 2026-06-24
**Author:** William Dennis
**Status:** Draft research notes (not implementation-ready)

---

## Overview

[Sim Francisco](https://simfrancisco.org) is a "distributionally accurate" synthetic
population of San Francisco — sampled from real US Census PUMS microdata — that can be
polled and perturbed with events to produce population-level predictions (elections,
ballot measures, approval ratings, prediction-market probabilities). Winner of 2nd place
at the Claude Opus 4.8 Build Day hackathon (May 2026).

This document maps structural parallels between Sim Francisco and our rl-health
interventions framework, identifies complementary components, and proposes a research
bridge.

---

## 1. Shared Sub-Problem: Population-Scale Decision Modeling

Both frameworks solve the same architectural challenge:
**how to model a heterogeneous population of agents without paying per-agent cost at inference time.**

| Dimension | Sim Francisco | rl-health-interventions |
|-----------|---------------|------------------------|
| Population size | 10,000 agents | Configurable (typically 50–1000 participants) |
| Population source | Real PUMS microdata (8 SF PUMAs) | NHANES step count + `SyntheticDataGenerator` (gaussian) |
| Inference compression | Cluster into ~12–160 archetypes, one batched LLM call per archetype | Per-participant bandit policy conditioned on state features |
| Expensive operation | LLM call (Claude Sonnet) | Posterior update / reward model evaluation |
| State representation | ValueVector (8 axes) + persona prose | Feature vectors g(s), f(s) — step counts, weather, time |

**The shared insight:** Neither system treats each individual independently at
inference. They compress the population into clusters or state discretisations, then
run expensive reasoning per compressed unit.

---

## 2. Architectural Parallels

### 2.1 Persona Construction ↔ Feature Engineering

**Sim Francisco pipeline:**
```
PUMS record → ValueVector(8 axes) → persona prose → LLM reads prose → p_yes
```
- ValueVector: economic L/R, social L/R, institutional trust, change-vs-status-quo,
  6 issue saliences
- Calibrated to SF electorate demographics (not hand-tuned)
- Persona prose = demographics + name + occupation + neighborhood + religion + hobbies
- LLM makes the actual probabilistic decision over the full prose

**rl-health pipeline:**
```
step_data → construct_heartsteps_features(g(s), f(s)) → Bayesian reward model → action
```
- g(s): 12 features (steps, time-of-day, day-of-week, weather, variability, averages)
- f(s): 8 treatment-effect features
- Bayesian posterior update selects actions based on reward model

**Common pattern:** Both maintain a compact structured representation of individual
state/tendency, and use it to condition a decision-making module. The structured
representation bridges between demographic/behavioral data and the decision policy.

### 2.2 Verification / Validation Loop

| Sim Francisco | rl-health-interventions |
|---|---|
| `rubric.yaml` — frozen ground-truth targets | `VERIFICATION_REPORT.md` + test suite |
| `validate` — exit 0 iff headline >= 0.85 | `cargo test` / `pytest` |
| Leakage-free backtest (knowledge-cutoff models) | Paper reproduction (HeartSteps V2) |
| Self-correction loop (`/goal`) | CI hook (`pre-push`) |
| Adversarial critic agent | Manual review |

**Key gap:** Sim Francisco's `rubric.yaml` pattern is a frozen evaluation that **cannot
be gamed** once committed. Our test suite verifies code behavior, but we don't have
a frozen behavioral target that proves "the simulation produces correct population-level
outcomes." This is a stronger guarantee.

### 2.3 Counterfactual / Causal Reasoning

**Sim Francisco:** Naive counterfactual support:
```python
run_counterfactual(pop, base_poll, event) → (baseline, with_event, delta)
```
It re-polls the population with a perturbed event and measures the shift. The result
is scored on **direction** (does it shift the right way?), not fabricated magnitude.

**rl-health:** The bandit policy optimises expected reward:
```
H(x, a) = expected next-state value after action a at dosage x
```
The ProxyValueFunction solves the Bellman equation on a discretised dosage grid.
This is the same causal question with a different estimator.

---

## 3. Research Bridge: Heterogeneous Agent Populations for Health Interventions

### The Problem

Current `SyntheticDataGenerator` produces participants from `N(8000, 2000)` step counts:
```python
"steps": rng.normal(8000, 2000, size=(n_users, n_timesteps)).astype(np.int64)
```
This ignores demographic heterogeneity entirely. Real populations have:
- Different baseline activity levels (age, health status, occupation)
- Different responsiveness to interventions (age, motivation, routine)
- Different availability patterns (work schedules, caregiving)

### The Bridge

Replace the gaussian generator with PUMS-conditioned sampling, mirroring Sim Francisco:

```
NHANES PUMS → demographic profile → ValueVector analogue
  (activity_level, health_motivation, routine_stability, risk_tolerance)
    → persona/context features → reward model parameters
```

This produces a synthetic population where:
- A 65yo retiree has different step baseline, availability, and intervention-response
  than a 23yo gig worker
- The reward model alpha/beta can be conditioned on demographic features
- The bandit policy learns **heterogeneous treatment effects** rather than averaged ones

### Research Question

> Can a bandit policy trained on demographically-heterogeneous synthetic populations
> outperform one trained on homogeneous (gaussian) populations when evaluated on
> real NHANES holdout data?

This is testable with our current framework + a richer generator.

---

## 4. Methodological Parallels Worth Adopting

### 4.1 Frozen Rubric Validation

Adopt Sim Francisco's `rubric.yaml` pattern for simulation evaluation:
- Fix ground-truth behavioral targets (e.g., "mean steps increase >= 5% with p<0.01")
- These targets are committed and immutable
- `validate` exits 0 only if targets met
- No tuning against the rubric allowed

### 4.2 Stratified Population Generation

Sim Francisco uses PUMS survey weights (`PWGTP`) for post-stratification. We could
use NHANES sampling weights similarly, ensuring the synthetic population matches
the joint distribution of age/BMI/activity-level in the target population.

### 4.3 Leakage-Free Temporal Backtesting

Sim Francisco's backtest uses GPT-4o (knowledge cutoff Oct 2023) to predict Nov 2024
outcomes, ensuring the model isn't memorising results. We can adopt the same pattern:
train NHANES reproduction on pre-2020 data, validate on post-2020 holdout to prove
the simulation doesn't leak temporal information.

### 4.4 Demographic Value Vectors

rl-health currently has no analogue to Sim Francisco's ValueVector for agent
"personality." Adding compact axes like (activity_motivation, routine_stability,
digital_literacy, health_literacy) to each agent would enable:
- Persona-conditional policies in contextual bandits
- Richer state representations beyond step counts + weather
- Sim Francisco-style "ask this agent anything" capability as a probe

---

## 5. Replication Potential

If you wanted to reproduce or extend Sim Francisco within our framework:

**Stack choice:** Rust (axum + sqlite) for the prediction engine, Rust+React for the
frontend because they can reproduce the population-level reproduction to the original
research. Substituting Rust here is also sufficient since our Python framework
can handle the same evaluation pipeline at smaller scale.

**Minimal viable reproduction:**
1. Source NHANES microdata (already in `paper_reproduction/data/`)
2. Build ValueVector assignment rules conditioned on demographics (the mapping between
   demographics and values is more compact than Sim Francisco's and could be simpler)
3. Add a prediction engine that conditions policy on persona context
4. Write a `rubric.yaml` with frozen holdout targets
5. Deploy via our existing GH Pages setup

**Key difference:** Sim Francisco's prediction target is a population-level election
outcome. Ours would be a population-level health outcome (step response to
notification). The estimation logic is structurally identical but the reward
computation uses health features instead of demographic voting patterns.

---

## 6. Open Questions

1. **Does demographic heterogeneity in the simulation improve downstream policy
   quality?** Or is the signal too weak relative to within-person variance?
2. **Can we use LLM-as-simulator for health behavior the way Sim Francisco
   uses it for political behavior?** Cost would be ~100x our current evaluation
   but could serve as a probe/exploration tool.
3. **Is the archetype clustering approach applicable to our state space?**
   Sim Francisco clusters demographics; we could cluster temporal behavioral
   patterns (weekday vs weekend, morning vs evening) for batching.
4. **Does the ValueVector generalize to health?** Economic/social L/R maps
   poorly to health behavior. What are the right axes?

---

## References

- Sim Francisco: https://simfrancisco.org
- GitHub: https://github.com/tejasprabhune/simfrancisco
- Claude blog post: https://claude.com/blog/meet-the-winners-of-our-claude-opus-4-8-build-day-hackathon
- Tejas Prabhune: Berkeley, GPU efficiency + user simulation for post-training
- HeartSteps V2 paper: Liao et al. (2019), arXiv:1909.03539
