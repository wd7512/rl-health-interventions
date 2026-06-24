---
title: "Decision Catalogue — State, Action & Reward Space Design"
status: "active"
date: "2026-06-23"
purpose: "Map all open design decisions for the rl-health-interventions MDP. The primary entry point for implementation."
related: "state-space-design/ · action-space-design/ · decision-trees/online-vs-offline-rl.md"
---

# Decision Catalogue — State, Action & Reward Space Design

> Every open design decision for the MDP state space, action space, and
> reward function in one place. For each decision: what needs deciding,
> what options have been considered, what the evidence says, and where to
> find the deep-dive evidence.
>
> **All decisions are currently open.** Nothing has been committed to. The
> catalogue maps the decision space — it does not resolve it.
>
> The upstream framing decision (simulation-based policy evaluation) is
> resolved separately in
> [decision-trees/online-vs-offline-rl.md](decision-trees/online-vs-offline-rl.md).

## Dependency diagram

```text
D14 (Phase 1 vs 2 deferral) ─── constrains timing of all
│
├─→ D1 (step encoding) ──→ D11 (reward design) ←── D8 (non-activity reward)
├─→ D2 (factored vs flat) ──→ D12 (algorithm class)
├─→ D3 (hidden state) ←→ D9 (mood/sleep: reward vs state)
├─→ D4 (trend dimension)          independent
├─→ D5 (time-of-day)              independent
├─→ D6 (day type)                  independent
└─→ D7 (action set) ──→ D8 (non-activity reward)
                       ──→ D10 (burden model) ←→ D3
                       ──→ D13 (evaluation strategy) ←── D8
```

## D1. Step count encoding & boundaries

**Status:** open
**Rationale:** Step encoding controls what the agent "knows" about user activity — too coarse misses clinical distinctions, too fine overfits to noise.

How should raw step counts be discretized or represented as part of the state vector?

| Option | Description | Evidence |
|--------|-------------|----------|
| 4 threshold bins (<4k / 4-7k / 7-10k / ≥10k) | Four clinically meaningful activity levels | Saint-Maurice 2020 JAMA Fig 2 — mortality RR plateaus at 6k-8k (older) / 8k-10k (younger); Paluch 2022 Lancet PH Fig 3 — 15% RR reduction per 1k steps, curve flattens above ~8k; StepCountJITAI sim (Karine 2024) — 4-bin > 2-bin for policy learning |
| Binary sedentary/active | Two-level classification | Klasnja 2019 HeartSteps V1 MRT — sedentary state predicts 2x nudge effectiveness; Liao 2019 HeartSteps V2 — binary active/last-30-min for contextual bandit |
| Continuous raw count | Unbinned step count | Gateno 2023 Health Gym — continuous + function approximation; no evidence continuous > binned for policy learning |

**Sub-questions:**

- Boundary values [parameter — sensitivity analysis]: 4k/7k/10k vs 3k/6k/9k vs quartile-based. 10k is pedometer marketing origin (1965). Lee 2022 JAMA Intern Med — minimal effective dose ~4k-4.5k for 50% of optimal benefit.
- Upper bin for engagement outcomes: All evidence is for mortality. Engagement curve unknown.
- Threshold gaming risk: 4 bins reduces it vs binary but does not eliminate.

→ Deep dive: [state-space-design/step-bin-evidence.md](state-space-design/step-bin-evidence.md)

## D2. Factored vs flat state representation

**Status:** open
**Rationale:** Determines whether state is a feature vector (contextual bandit / function approximation) or enumerated index (tabular). Constrains algorithm selection.

Should the state be represented as a factored feature vector or a flat discrete index?

| Option | Description | Evidence |
|--------|-------------|----------|
| Factored feature vector | Each state dimension is an independent feature | Trella 2022 Sec 3.1 — avoid combinatorial explosion via function approximation; HeartSteps V2 Liao 2019 — feature vector + Thompson sampling, 34% step increase vs random |
| Flat discrete state index | State is an enumerated composite key | StepCountJITAI sim (Karine 2024) — tabular 20-40 states; HeartSteps V1 MRT — 2x2x2=8 strata |
| Hybrid | Partial factoring | No published system uses this |

**Sub-questions:**

- Migration path: codebase currently uses flat state keys in per_step_reward
- Cardinality: 4x4x2x3x?x? factorial. Tractable with factored, intractable flat.

**Constrains:** D12

→ Deep dive: [state-space-design/reference-configs.md](state-space-design/reference-configs.md)

## D3. Hidden psychosocial state variables

**Status:** open
**Rationale:** Mood/stress/sleep could modify how users respond to interventions, but no deployed RL system models them as state. The only MRT that tested stress as moderator found null.

Should mood, stress, or sleep be included in the state representation?

| Option | Description | Evidence |
|--------|-------------|----------|
| Exclude from MVP | Do not include psychosocial variables | Rabbi 2019 Depression MRT — stress tested as moderator of nudge→activity, null result; no deployed RL-for-health system includes mood/stress as state variable |
| Include as observable context feature | Add as available context features when measured | Trella 2022 — recommends stress as optional context feature if available; 1 MRT (Rabbi 2019) tested stress, null |
| Include as hidden latent state | Model as POMDP hidden state | No published precedent; POMDP structure; purely speculative |

**Sub-questions:**

- Sleep vs mood/stress: stronger observational evidence (multiple wearable studies) but equally weak causal moderation

**Cross-cutting:** D9 (same variables, opposite framing), D10 (fatigue may collapse into mood state)

→ Deep dive: [state-space-design/hidden-state-evidence.md](state-space-design/hidden-state-evidence.md)

## D4. Trend dimension design

**Status:** open
**Rationale:** No published RL system has used a trend dimension. This is novel and untested.

Should a trend dimension (increasing/stable/decreasing activity) be included in the state?

| Option | Description | Evidence |
|--------|-------------|----------|
| 3-level categorical increasing/stable/decreasing | Three-bin trend classification | No precedent |
| Continuous slope | Raw slope value | No precedent |
| Exclude from MVP | No trend dimension | Consistent with all 6 reference systems |

**Sub-questions:**

- Computation method [parameter]: rolling OLS, simple differencing, binary sign of 3-day diff, EMA. No literature basis.
- Window size [parameter — sensitivity analysis]: 3-7 days. No literature basis.
- Interaction with step bins: intuitive but untested

→ Deep dive: [state-space-design/reference-configs.md](state-space-design/reference-configs.md)

## D5. Time-of-day encoding

**Status:** open
**Rationale:** Time-of-day is strongest moderator in HeartSteps V2, but optimal granularity unknown.

How should time-of-day be encoded in the state?

| Option | Description | Evidence |
|--------|-------------|----------|
| 2 bins AM/PM | Binary morning/afternoon | Klasnja 2019 — binary time-of-day MRT stratification |
| 4 bins | Quarter-day divisions | No direct evidence |
| 5 bins decision points | Aligned with HeartSteps delivery schedule | Liao 2019 — time-of-day strongest moderator; 5 decision points/day deployed |
| Continuous | Unbinned time value | No precedent in domain |

→ Deep dive: [state-space-design/reference-configs.md](state-space-design/reference-configs.md)

## D6. Day type encoding

**Status:** open
**Rationale:** Trivial in isolation, but contributes to state cardinality in flat representation.

How should day type (weekday/weekend) be encoded?

| Option | Description | Evidence |
|--------|-------------|----------|
| Binary weekday/weekend | Two-level classification | Universal across all 6 reference systems |
| 3-level weekday/weekend/holiday | Include holidays | No precedent |
| Exclude | No day type dimension | No system has tested |

→ Deep dive: [state-space-design/reference-configs.md](state-space-design/reference-configs.md)

## D7. Action set composition

**Status:** open
**Rationale:** Most consequential decision — highest out-degree in dependency diagram. Determines what the agent can do.

What actions should the agent be able to select?

| Option | Description | Evidence |
|--------|-------------|----------|
| Activity-only nudges + idle (3-4 actions) | Step-focused nudges plus no-op | HeartSteps V2 binary — 34% step increase vs random; StepCountJITAI 2-4 actions — burden-aware > burden-agnostic; Trella 2022 prescribes 2-6 actions |
| Mixed activity + non-activity + idle (6 proposed) | Activity nudges, journaling, sleep, social, idle | Non-activity interventions have strong standalone RCT evidence (Smyth 2018 PAJ, Koffel 2018 sleep); no published RL system includes them in action set; Trella 2022 — each action should map to distinct behavioral construct |
| Binary deliver/no-deliver (2 actions) | Simplest possible | HeartSteps V2 precedent; fastest convergence |
| Activity-only with intensity levels | Vary intensity not type | StepCountJITAI intensity variation; no comparison vs type-variation |

**Sub-questions:**

- gentle_nudge vs goal_nudge separation: JustWalk protocol uses goal-based feedback; HeartSteps generic suggestions; no RL comparison
- Social encouragement inclusion: Crum 2017 RCT d=0.32; Michie 2011 meta-analysis OR=1.2; HeartSteps secondary analysis null
- Idle/no-op inclusion: Trella prescribes; 4/6 reference systems include; universal consensus
- Behavioral construct validity: gentle_nudge and goal_nudge overlap motivation vs goal-setting

**Constrains:** D8, D10, D13

→ Deep dives: [action-space-design/reference-configs.md](action-space-design/reference-configs.md), [action-space-design/non-activity-interventions.md](action-space-design/non-activity-interventions.md)

## D8. Non-activity action reward

**Status:** open
**Rationale:** Non-activity actions do not increase step count. Under step-only reward, agent learns to avoid them.

How should non-activity actions (journaling, sleep hygiene, social) be rewarded?

| Option | Description | Evidence |
|--------|-------------|----------|
| Placeholder mood/sleep proxy signal | Reward based on mood/sleep improvement proxy | Smyth 2018 JMIR — journaling reduces depression/anxiety; Koffel 2018 — sleep hygiene improves PSQI; no study has used as proxy reward in RL |
| Defer to Phase 2 | Do not include non-activity actions in Phase 1 | RCTs show causal effect on mood/sleep but no sequential RL evidence; safest given evidence gap |
| Post-hoc evaluation only | Agent avoids them → selection frequency ~0 → uninformative | No precedent |
| Zero step reward + burden penalty (default) | Rational agent never selects them | StepCountJITAI, HeartSteps V2 — no step reward for non-activity |

**Blocked by:** D7

→ Deep dive: [action-space-design/non-activity-interventions.md](action-space-design/non-activity-interventions.md)

## D9. Mood/sleep: reward signal vs state variable

**Status:** open
**Rationale:** Same variables as D3, approached from opposite end. Cannot be both without double-counting.

Should mood/sleep be used as a reward signal or a state variable?

| Option | Description | Evidence |
|--------|-------------|----------|
| Reward signal (Phase 2 multi-objective) | Use mood/sleep improvement as reward component | Smyth 2018 PAJ — journaling reduces depression/anxiety; Koffel 2018 — sleep hygiene improves PSQI; consistent with Trella recommendation |
| State variable (latent) | Include as hidden state | Rabbi 2019 null result for stress; confounded observational evidence; no moderation evidence in MRTs |
| Exclude from MVP | Do not include | Consistent with D3 exclusion |

**Cross-cutting:** D3, D8

→ Deep dive: [state-space-design/hidden-state-evidence.md](state-space-design/hidden-state-evidence.md)

## D10. Burden/fatigue model

**Status:** open
**Rationale:** Cost side of reward. Per-action burden and cumulative fatigue may be same or different constructs. All values are heuristic.

How should action burden and cumulative fatigue be modelled?

| Option | Description | Evidence |
|--------|-------------|----------|
| Linear accumulator with decay | Per-action cost with temporal decay | Karine 2024 StepCountJITAI — b(a)=0.1-0.5, d=0.2, daily reset; burden-aware > burden-agnostic |
| Static per-action cost | Fixed cost per action type | Trella 2022 — burden should be reflected in reward; no specific values |
| No explicit model | Frequency limited by decision points | HeartSteps V2 — no explicit penalty; frequency limited by 5 decision points/day |

**Sub-questions:**

- Fatigue vs burden: same construct? [decision]. StepCountJITAI collapses; Trella mentions fatigue conceptually; no system separates
- Penalty magnitudes [parameter — sensitivity analysis]: 0.0-0.25 proposed vs 0.1-0.5 StepCountJITAI. No empirical basis (action-burden-evidence.md)
- Decay rate [parameter — sensitivity analysis]: d=0.2 StepCountJITAI. Generalizability unknown
- Reset schedule [decision]: daily (StepCountJITAI) vs weekly vs continuous-only

**Cross-cutting:** D3 (if mood hidden, fatigue may collapse into it), D7

→ Deep dive: [action-space-design/action-burden-evidence.md](action-space-design/action-burden-evidence.md)

## D11. Reward function design

**Status:** open
**Rationale:** Overall structure of R(s,a,s'). Proposed 3-term form but base reward and time horizon unresolved.

What should the reward function look like?

| Option | Description | Evidence |
|--------|-------------|----------|
| R = base_step_reward - burden - fatigue | Three-term form (proposed) | Trella 2022 + StepCountJITAI implement this form; no system validates fatigue term |
| R = step_count_reward only | HeartSteps form | Liao 2019 — 30-min post-decision step count increase |
| R = step_count_reward - burden only | Two-term, no fatigue | Equivalent to collapsed D10 |
| Multi-objective (steps + mood + sleep) | Multiple reward components | RCTs show causal effects; no RL implementation; deferred to Phase 2 |

**Sub-questions:**

- Base reward definition [decision]: raw count / bin categorical / binary threshold / 30-min window. No published comparison.
- Reward time horizon [decision]: 30-min post-decision / same-day cumulative / next-day / end-of-episode. HeartSteps uses 30-min; longer horizons more clinically meaningful but delay learning.

**Blocked by:** D1, D8

→ Deep dives: [action-space-design/reference-configs.md](action-space-design/reference-configs.md), [action-space-design/action-burden-evidence.md](action-space-design/action-burden-evidence.md)

## D12. Algorithm class (contextual bandit vs full RL)

**Status:** open
**Rationale:** Determines whether agent is myopic (1-step) or plans over horizon. Affects state representation and what agent can learn.

What algorithm class should be used for policy learning?

| Option | Description | Evidence |
|--------|-------------|----------|
| Contextual bandit (Thompson sampling) | Myopic, no transition model | Liao 2019 HeartSteps V2 — most successful deployed RL-for-health; 34% step increase; state as features; no transition model |
| Tabular RL with known transitions | Full RL, discrete state/action | Karine 2024 StepCountJITAI — tabular Q with rule-based transitions; optimal policy if transitions accurate |
| Deep RL with function approximation | Full RL, continuous state | Gateno 2023 Health Gym — continuous state; no published success in deployed health RL |

**Blocked by:** D2. Note: overall framing resolved in decision-trees/online-vs-offline-rl.md.

## D13. Non-activity evaluation strategy

**Status:** open
**Rationale:** If non-activity actions included with no direct step reward, how to evaluate effectiveness?

How should non-activity actions be evaluated if they don't contribute to step reward?

| Option | Description | Evidence |
|--------|-------------|----------|
| Post-hoc trajectory analysis | Analyse trajectories after training | No precedent; selection frequency ~0 makes this uninformative |
| A/B test vs activity-only agent | Compare two agent variants | No precedent |
| MRT experimental arms | Embed in micro-randomized trial | No precedent in RL |
| Simulate with assumed transition model | Use simulation to evaluate | No empirical basis |

**Blocked by:** D7, D8

→ Deep dive: [action-space-design/non-activity-interventions.md](action-space-design/non-activity-interventions.md)

## D14. Phase 1 vs Phase 2 deferral

**Status:** open
**Rationale:** Meta-decision. Not every decision needs Phase 1 resolution.

Which decisions must be resolved for Phase 1 and which can be deferred?

| Decision | Phase 1 critical? | Reasoning |
|----------|-------------------|-----------|
| D1 step encoding | Likely yes | Core representation |
| D2 factored/flat | Likely yes | Architecture-level |
| D3 hidden state | Likely no | No evidence; speculative |
| D4 trend | Likely no | No precedent |
| D5 time-of-day | Likely yes | Required for state |
| D6 day type | Likely yes | Trivial; binary default |
| D7 action set | Likely yes | Core representation |
| D8 non-activity reward | Depends on D7 | If D7 includes non-activity |
| D9 reward vs state | Likely no | Phase 2 |
| D10 burden/fatigue | Likely yes | Needed for reward |
| D11 reward design | Likely yes | Core MDP |
| D12 algorithm class | Likely yes | Core architecture |
| D13 evaluation strategy | Depends on D7/D8 | If non-activity in Phase 1 |
| D14 this decision | Yes (meta) | Must be resolved first |

> This mapping is a proposed starting point, not a resolution. D14 is open.

---

## Summary

| # | Decision | Status | Key evidence strength | Deep-dive file |
|---|----------|--------|----------------------|----------------|
| D1 | Step count encoding | open | Strong | step-bin-evidence.md |
| D2 | Factored vs flat | open | Moderate | reference-configs.md |
| D3 | Hidden psychosocial state | open | Weak | hidden-state-evidence.md |
| D4 | Trend dimension | open | None | reference-configs.md |
| D5 | Time-of-day encoding | open | Moderate | reference-configs.md |
| D6 | Day type encoding | open | Moderate | reference-configs.md |
| D7 | Action set composition | open | Moderate | reference-configs.md, non-activity-interventions.md |
| D8 | Non-activity reward | open | None | non-activity-interventions.md |
| D9 | Reward vs state | open | Strong/Weak | hidden-state-evidence.md, non-activity-interventions.md |
| D10 | Burden/fatigue model | open | Weak | action-burden-evidence.md |
| D11 | Reward function design | open | Moderate | reference-configs.md, action-burden-evidence.md |
| D12 | Algorithm class | open | Moderate | reference-configs.md |
| D13 | Evaluation strategy | open | None | non-activity-interventions.md |
| D14 | Phase 1/2 deferral | open | N/A | — |

---

## Conventions

- **Status** is one of: `open` / `closed` / `deferred`
- **Evidence references** follow the pattern: Author Year Journal Figure/Table — specific finding
- **Sub-questions** tagged `[parameter — sensitivity analysis]` are continuous-valued choices for sensitivity analysis
- **Deep-dive links** point to files in `state-space-design/` or `action-space-design/`
- **Upstream framing decision**: [decision-trees/online-vs-offline-rl.md](decision-trees/online-vs-offline-rl.md)
