---
title: "Hidden State Variables — Mood, Stress, and Sleep as Latent MDP Variables"
status: "draft v0.1"
date: "2026-06-23"
purpose: "Review whether the literature supports modelling mood/stress/sleep as latent state variables in an activity RL system"
---

# Hidden State Variables — Mood, Stress, and Sleep as Latent MDP Variables

> This file reviews whether mood, stress, or sleep have been (a) modelled
> as state variables in deployed RL systems, (b) shown to causally moderate
> nudge effectiveness in MRTs, or (c) linked to activity transitions in
> observational studies.

## 1. In deployed RL systems

No deployed RL-for-health system has modelled mood, stress, or sleep as a
state variable.

- **HeartSteps V2** uses only observable features (time, location, weather,
  recent activity). No mood/stress state.
- **StepCountJITAI simulation** uses activity + time features only.
- **Health Gym** uses step count metrics + heart rate (wearable), but not
  mood/stress.

**Reference configs**: See `reference-configs.md` in this directory.

## 2. In MRTs (causal moderation evidence)

| Study | Variables | Finding |
|-------|-----------|---------|
| Klasnja (2019, HeartSteps V1 MRT) | Recent activity (binary) | Sedentary state → suggestion more effective |
| Liao (2019, HeartSteps V2) | Time, location, weather, activity | Time-of-day strongest moderator of nudge effect |
| Rabbi (2019, Depression MRT) | Stress (EMA-reported) | No significant moderation of nudge → activity pathway |

**The only MRT that tested stress as a moderator found no significant
effect.** This is a single study with limited power, but it is the best
available evidence.

## 3. Observational evidence (correlational, not causal)

| Association | Evidence strength | Direction |
|-------------|------------------|-----------|
| Poor sleep → lower next-day activity | Strong (multiple wearable studies) | Consistent |
| High stress → lower activity | Moderate (self-report studies) | Consistent |
| Positive mood → higher activity | Weak (most studies cross-sectional) | Consistent but confounded |

**Problem**: All observational evidence is confounded by reverse causation
(more activity → better mood) and common causes (weather, illness, schedule).

## 4. Could mood/stress/sleep be an *action outcome* rather than a state?

An alternative framing: mood/stress/sleep are not state variables but
**reward signals** that non-activity actions (journaling, sleep hygiene)
can affect. This is more consistent with the literature:

- PAJ trial: journaling → reduced depression/anxiety (outcome measure)
- Sleep interventions: sleep hygiene → better sleep quality (outcome measure)
- No study shows these variables *moderate* the effect of *other* actions

**Implication for design**: Mood/stress/sleep could serve as:
- A **separate reward term** (Phase 2 multi-objective), or
- A **context feature** if measured via EMA (requires real-time data),
  but not as a hidden transition-modifying state

## 5. Summary

| Use case | Evidence | Recommendation |
|----------|----------|---------------|
| As a hidden state affecting transitions | No evidence | Model only if you hypothesise this mechanism; it is not grounded in literature |
| As a context feature in a contextual bandit | Weak (1 MRT tested, null result) | Speculative — include as experimental condition |
| As a separate reward signal (multi-objective) | Strong (RCTs show causal effect on these outcomes) | Defer to Phase 2 |

**Bottom line**: The literature does not support modelling mood/stress as
a hidden state variable in the MVP. If included, it would be speculative
and should be flagged as such.
