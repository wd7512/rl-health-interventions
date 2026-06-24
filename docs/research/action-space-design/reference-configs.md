---
title: "Reference Configurations — Action Space"
status: "draft v0.1"
date: "2026-06-23"
purpose: "Describe what 5+ published systems have done for action space design"
---

# Reference Configurations — Action Space

> Descriptive, not prescriptive. Each entry reports what a published system
> actually implemented. No judgement, no recommendation.

## 1. HeartSteps V2 — Liao et al. (2019, ACM IMWUT)

| Component | Detail |
|-----------|--------|
| **System type** | RL in a live MRT (90 days, 5 decision points/day) |
| **Action set** | `deliver_suggestion` / `no_suggestion` (binary, 2 actions) |
| **Action content** | Contextually-tailored activity suggestions (walking, stretching) |
| **Burden/cost** | Implicit — fewer delivered actions = lower burden. No explicit penalty. |
| **Reward** | Activity increase in the 30-minute window after the decision point |
| **RL algorithm** | Contextual Thompson sampling |
| **Context features** | Time-of-day, location, weather, recent activity, day of week |
| **Key result** | 34% increase in step counts vs uniform random delivery |

**Relevance**: Most directly comparable system. Binary action set is the minimal
case — suggests small action spaces are sufficient for measurable effects.

## 2. HeartSteps V1 — Klasnja et al. (2019, Annals of Behavioral Medicine)

| Component | Detail |
|-----------|--------|
| **System type** | Micro-randomized trial (pre-RL, 6 weeks) |
| **Action set** | `suggestion` / `no_suggestion` (binary, 2 actions) |
| **Action content** | Walking suggestions vs generic activity suggestions |
| **Burden/cost** | Not modeled — MRT randomisation, no learning |
| **Reward** | Activity within 30 minutes (accelerometer) |
| **Transition** | None — pure randomised assignment |
| **Key finding** | Suggestions delivered when user is sedentary are significantly more effective |

**Relevance**: Established that timing (state context) matters more than message
content. Direct evidence that state-aware delivery outperforms state-agnostic.

## 3. Trella et al. (2022, Algorithms) — Design Guidelines

| Component | Detail |
|-----------|--------|
| **System type** | Pre-implementation design framework (not an experiment) |
| **Action set** | Proposes 2–6 actions as a rule of thumb for online learning |
| **Action content** | Each action should map to a distinct behavioural construct |
| **Burden/cost** | Introduces burden as a concept; advises including idle/no-op |
| **Reward** | Not specified — framework-level |
| **Transition** | Not specified — framework-level |
| **Key prescription** | Small action spaces learn faster with limited data; actions should vary in intensity/cost |

**Relevance**: Design principles used in the current MVP action set and the
proposed 6-action expansion. Not an empirical result — expert consensus.

## 4. Positive Affect Journaling (PAJ) — Smyth et al. (2018, JMIR Mental Health)

| Component | Detail |
|-----------|--------|
| **System type** | RCT, not an RL system (3 sessions/week × 12 weeks) |
| **Action set** | `journaling_prompt` (single intervention arm vs control) |
| **Action content** | Write about positive experiences for 20 minutes |
| **Burden/cost** | Modest — 20-minute writing sessions, 3×/week |
| **Outcome (reward proxy)** | Reduced anxiety/depression, improved well-being (PHQ-9, GAD-7) |
| **Transition** | Pre-post within-person — no sequential decision-making |
| **Key result** | Clinically meaningful reduction in depression and anxiety scores |

**Relevance**: Evidence that journaling is an effective single intervention.
Its delivery mechanism (notification → in-app writing session) is compatible
with the notification-based RL framework.

## 5. StepCountJITAI Simulation — Karine et al. (2024)

| Component | Detail |
|-----------|--------|
| **System type** | Simulation study using HeartSteps-derived parameters |
| **Action set** | 2–4 actions depending on simulation condition |
| **Action content** | Activity suggestions at varying intensities |
| **Burden/cost** | Models user burden as a function of suggestion frequency |
| **Reward** | Simulated step count based on transition probabilities |
| **Transition** | Rule-based, parameterised from HeartSteps data |
| **Key finding** | Smaller action sets (2–4) converge faster; burden-aware policies outperform burden-agnostic |

**Relevance**: Simulation methodology closest to our experiment runner. Burden
model integration shows feasibility of action-cost-aware reward.

## 6. JustWalk Protocol — JMIR Res Protoc (2023)

| Component | Detail |
|-----------|--------|
| **System type** | Protocol for a goal-setting + reminder walking intervention |
| **Action set** | `goal_reminder` / `no_reminder` (binary) + step goal feedback |
| **Action content** | Personalised step goals + progress feedback messages |
| **Burden/cost** | Not explicitly quantified — low-burden by design |
| **Reward** | Step count achievement (meeting daily goal) |
| **Transition** | Not applicable — protocol-only, no implementation |
| **Relevance** | Informs `goal_nudge` action design — goal-setting is distinct from generic motivational prompts |

## Summary: Action cardinality across systems

| System | Actions | Includes idle? | Includes non-activity? |
|--------|---------|----------------|----------------------|
| HeartSteps V2 | 2 | Yes (no_suggestion) | No |
| HeartSteps V1 | 2 | Yes (no_suggestion) | No |
| Trella guidelines | 2–6 | Prescribed | Not specified |
| PAJ | 1 (trial arm) | No (control group) | Yes (journaling) |
| StepCountJITAI | 2–4 | Yes | No |
| JustWalk protocol | 2–3 | Yes | No |

**Observation**: No published RL system has used non-activity actions
(journaling, sleep hygiene) as part of its action set. Their inclusion in the
proposed set is novel — the evidence supports them as *effective standalone
interventions*, not as RL actions in a multi-action policy.
