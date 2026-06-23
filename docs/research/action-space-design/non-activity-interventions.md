---
title: "Non-Activity Interventions — Evidence Review"
status: "draft v0.1"
date: "2026-06-23"
purpose: "Review causal evidence for journaling, sleep hygiene, and social encouragement as intervention actions"
---

# Non-Activity Interventions — Evidence Review

> This file collects evidence for three non-activity intervention types that
> could be included in the action set. The emphasis is on *causal* evidence
> (RCTs, MRTs) and mechanism (how the intervention affects the MDP dynamics),
> not correlation.

## 1. Journaling (Expressive Writing / Positive Affect Journaling)

### Causal evidence

| Study | Design | N | Finding | Causal claim strength |
|-------|--------|---|---------|----------------------|
| Smyth (1999, JAMA) | RCT, 4×20min sessions | 122 | Fewer physician visits, improved immune function | Strong — blinded RCT, objective outcomes |
| Norman (2004, Ann Behav Med) | RCT, expressive writing in chronic illness | 107 | Improved lung function (FEV1), reduced disease activity | Strong — objective biomarkers |
| Smyth (2018, JMIR Ment Health) | RCT, 3×/week × 12 weeks, online | 70 | Reduced anxiety/depression (PHQ-9, GAD-7) | Strong — online delivery matches notification framework |
| Baikie & Wilhelm (2005, Adv Psych Treatment) | Systematic review | ~20 studies | Immune, blood pressure, GP visits, lung/liver function improvements | Moderate — review, heterogeneous outcomes |

### Mechanism for MDP dynamics

Journaling could affect the RL system through three pathways:

1. **Burden reduction**: Writing about positive experiences may reduce
   perceived burden of subsequent activity prompts. *Not directly tested*.
2. **Mood → activity moderation**: Improved mood may increase the probability
   that a subsequent nudge leads to activity. *No MRT has tested this pathway.*
3. **Independent reward**: Journaling produces measurable psychological
   outcomes (mood, stress) that could serve as a secondary reward signal
   independent of step count.

**Evidence gap**: No published study has tested whether journaling *moderates*
the effect of a subsequent activity nudge. The mechanism is plausible
(stress → barrier to activity) but untested in a sequential decision-making
framework.

## 2. Sleep hygiene

### Causal evidence

| Study | Design | N | Finding | Causal claim strength |
|-------|--------|---|---------|----------------------|
| Koffel (2018, J Clin Sleep Med) | Brief sleep hygiene intervention RCT | 56 | Improved sleep quality (PSQI) | Strong — brief intervention (feasible for notification delivery) |
| Irwin (2016, Lancet) | Meta-analysis of sleep interventions | 66 studies | Moderate effect on sleep quality (g=0.43) | Strong — meta-analytic |
| Baron (2013, Sleep Med Rev) | Review of mHealth sleep interventions | 12 studies | Digital sleep interventions effective but engagement drops | Moderate — engagement is the barrier |

### Mechanism for MDP dynamics

Sleep → next-day activity link:
- Consistent cross-sectional evidence: poor sleep → lower next-day activity
- Longitudinal evidence is weaker: few studies wearables measure both sleep
  and activity at daily resolution
- Causal pathway (sleep hygiene → better sleep → more next-day activity) has
  not been tested in an MRT framework

**Evidence gap**: The sleep-activity pathway is well-established
observationally but has not been tested as a *moderator* of nudge
effectiveness in a sequential intervention framework.

## 3. Social encouragement

### Causal evidence

| Study | Design | N | Finding | Causal claim strength |
|-------|--------|---|---------|----------------------|
| Crum (2017, Health Psychol) | RCT of social vs informational messages | 128 | Social messages increased activity more (d=0.32) | Moderate — single study |
| Michie (2011, Ann Behav Med) | BCT taxonomy meta-analysis | 101 studies | Social support (unspecified) is a weak BCT (OR=1.2) | Weak — broad category, heterogeneous |
| HeartSteps content variants | MRT secondary analysis | 40 | Social comparison messages no better than informational | Weak — not significant |

### Mechanism for MDP dynamics

Social encouragement may operate through a different pathway than
informational nudges:
- **Distinct behavioural construct** (per Trella 2022 requirement for action
  separation)
- May affect *self-efficacy* rather than *direct motivation*
- Evidence is weaker than for journaling or sleep hygiene

## Summary: Evidence strength by mechanism

| Mechanism | Journaling | Sleep hygiene | Social encouragement |
|-----------|-----------|---------------|---------------------|
| Independent outcome effect | Strong (RCTs) | Strong (RCTs) | Weak-to-moderate |
| Burden reduction | Untested | Untested | Untested |
| Nudge moderation (mood → activity) | Untested | Untested | Untested |
| Independent reward signal | Feasible (mood/stress) | Feasible (sleep quality) | Unclear |

## Implication for reward design

None of the three mechanisms have sufficient evidence to parameterise a
causal model for the MVP. The literature supports them as *effective
standalone interventions* but not as *moderators of sequential nudge
efficacy*. This means:

- Including them in the action set with a step-only reward will cause the
  agent to avoid them (no step benefit + burden penalty)
- A placeholder reward signal (mood/sleep proxy) would be speculative but
  necessary if they are to be learned in the MVP
- Deferring to Phase 2 (multi-objective reward) is the safest option given
  the evidence strength

See `action-burden-evidence.md` for how existing systems handle the
burden/fatigue side of non-activity actions.
