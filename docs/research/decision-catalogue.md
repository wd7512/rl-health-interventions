---
title: "Decision Catalogue — State, Action & Reward Space Design"
status: "active"
date: "2026-06-23"
purpose: "Map all open design decisions for the rl-health-interventions MDP. The primary entry point for implementation."
related: "state-space-design/ · action-space-design/ · decision-trees/online-vs-offline-rl.md"
---

# Decision Catalogue — State, Action & Reward Space Design

> Registry of all MDP design decisions and their resolution status.
> Resolved decisions documented in [resolved-decisions-sprint-1.md](resolved-decisions-sprint-1.md).

## Dependency diagram

```text
┌──────────────────────────────────────────────────────┐
│  STATE                                                │
│  D3 ←→ D9    D4    D5    D6                          │
│  D7 ──→ D8                                           │
├──────────────────────────────────────────────────────┤
│  REWARD                                               │
│  D1   D8   D9 ──→ D11                                │
├──────────────────────────────────────────────────────┤
│  ALGORITHM & EVALUATION                               │
│  D2 → D12    D10 ←→ D3    D13 ←── D8                │
│  D7 → D10              D7 → D13                      │
└──────────────────────────────────────────────────────┘
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

**Status:** closed (resolved: sleep ∈ {good, poor} included as state dimension; mood/stress excluded)
**Rationale:** Sleep quality (good/poor) is included as a binary state dimension that moderates within-day transition dynamics and also contributes to reward (see D9, D11). Mood and stress are excluded — Rabbi 2019 tested stress as moderator with null result, and no deployed RL-for-health system includes them as state variables.

**Resolution:**
- Sleep: 2 bins (good / poor), LLM-judged at day boundary
- Mood/stress: excluded from Sprint 1

**Key evidence:** Rabbi 2019 (stress null) · Trella 2022 (optional context) · Irwin 2016 (sleep interventions g=0.43)

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
| Defer to next sprint | Do not include non-activity actions in current sprint | RCTs show causal effect on mood/sleep but no sequential RL evidence; safest given evidence gap |
| Post-hoc evaluation only | Agent avoids them → selection frequency ~0 → uninformative | No precedent |
| Zero step reward + burden penalty (default) | Rational agent never selects them | StepCountJITAI, HeartSteps V2 — no step reward for non-activity |

**Blocked by:** D7

→ Deep dive: [action-space-design/non-activity-interventions.md](action-space-design/non-activity-interventions.md)

## D9. Mood/sleep: reward signal vs state variable

**Status:** closed (partially resolved: sleep is both state and reward — dual-role pattern; mood-only deferred to Phase 2)
**Rationale:** Sleep follows the same dual-role pattern as step_bin — it appears as a state dimension (D3 — transition moderator) and also contributes directly to reward (D11 — reward component). These roles capture distinct constructs: state role moderates transition dynamics, reward role values good sleep as an outcome. Mood-only remains deferred to Phase 2.

**Resolution:**
- Sleep: both state variable and reward signal (same pattern as step_bin)
- Mood: deferred to Phase 2 as potential reward component

**Key evidence:** Smyth 2018 PAJ (journaling reduces depression/anxiety) · Koffel 2018 (sleep hygiene improves PSQI) · Trella 2022 (mood as optional reward component)

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

**Status:** closed (resolved: R = α·f(step_bin') + (1-α)·g(sleep') − λ·𝟙[action≠idle]; f={inactive:0.0, moderate:0.5, active:1.0}, g={good:+1.0, poor:−1.0}, α=0.9, λ=0.05)
**Rationale:** Reward combines step count and sleep quality outcomes with a small penalty for non-idle actions to discourage spamming. Burden cost is expressed through reduced future activity probability (not subtracted directly from reward).

**Resolution:**
- Step reward: f maps post-transition step_bin' (inactive→0.0, moderate→0.5, active→1.0)
- Sleep reward: g maps post-transition sleep' (good→+1.0, poor→−1.0)
- α = 0.9 weights steps vs sleep (sweepable)
- λ = 0.05 per non-idle action
- 30-min immediate per-step horizon

**Key evidence:** Trella 2022 + StepCountJITAI (3-term form precedent) · Liao 2019 (30-min HeartSteps horizon) · Lee 2022 / Saint-Maurice 2020 (step-clinical thresholds for reward mapping)

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

## D14. Current sprint vs. next sprint deferral

**Status:** open
**Rationale:** Meta-decision. Not every decision needs current sprint resolution.

Which decisions must be resolved for the current sprint and which can be deferred?

| Decision | Current sprint critical? | Reasoning |
|----------|-------------------|-----------|
| D1 step encoding | Likely yes | Core representation |
| D2 factored/flat | Likely yes | Architecture-level |
| D3 hidden state | Likely no | No evidence; speculative |
| D4 trend | Likely no | No precedent |
| D5 time-of-day | Likely yes | Required for state |
| D6 day type | Likely yes | Trivial; binary default |
| D7 action set | Likely yes | Core representation |
| D8 non-activity reward | Depends on D7 | If D7 includes non-activity |
| D9 reward vs state | Likely no | Next sprint |
| D10 burden/fatigue | Likely yes | Needed for reward |
| D11 reward design | Likely yes | Core MDP |
| D12 algorithm class | Likely yes | Core architecture |
| D13 evaluation strategy | Depends on D7/D8 | If non-activity in current sprint |
| D14 this decision | Yes (meta) | Must be resolved first |

> This mapping is a proposed starting point, not a resolution. D14 is open.

---

## D15. Feature selection (PEARL)

**Status:** closed
**Rationale:** PEARL uses an XGBoost feature importance model (Figure 12) to select state features. This ADR documents the threshold choice and resulting feature set.

How many and which features should compose the state vector for PEARL-matched configs?

| Option | Description | Evidence |
|--------|-------------|----------|
| weight >= 0.5 (8 features) | Natural break: next weight is 0.32 (47% drop) | Lee 2025 arXiv Fig 12 — XGBoost model weights, 7,711 participants, 60 days |
| weight >= 0.4 (9 features) | More inclusive, adds one more dynamic feature | Sensitivity analysis: captures ~95% of cumulative importance |
| weight >= 0.6 (6 features) | Conservative, excludes two static features | Loses demographic coverage |

**Selected:** weight >= 0.5 — 8 features:

| Feature | Type | Weight |
|---------|------|--------|
| baseline_steps_mean | Static | 1.23 |
| weight | Static | 1.19 |
| age | Static | 1.17 |
| baseline_walk_pattern | Static | 1.13 |
| baseline_steps_std | Static | 1.02 |
| recent_steps_mean | Dynamic | 0.88 |
| recent_walk_pattern | Dynamic | 0.70 |
| morning_steps_ratio | Dynamic | 0.63 |

**Sub-questions:**

- Sensitivity analysis on threshold: 0.4 vs 0.5 vs 0.6
- Static features used only for persona assignment (fixed per episode)
- Dynamic features updated each step from transition tables

---

## Summary

| # | Decision | Status | Key evidence strength | Deep-dive file |
|---|----------|--------|----------------------|----------------|
| D1 | Step count encoding | open | Strong | step-bin-evidence.md |
| D2 | Factored vs flat | open | Moderate | reference-configs.md |
| D3 | Hidden psychosocial state | closed | Weak | hidden-state-evidence.md |
| D4 | Trend dimension | open | None | reference-configs.md |
| D5 | Time-of-day encoding | open | Moderate | reference-configs.md |
| D6 | Day type encoding | open | Moderate | reference-configs.md |
| D7 | Action set composition | open | Moderate | reference-configs.md, non-activity-interventions.md |
| D8 | Non-activity reward | open | None | non-activity-interventions.md |
| D9 | Reward vs state | closed | Strong/Weak | hidden-state-evidence.md, non-activity-interventions.md |
| D10 | Burden/fatigue model | open | Weak | action-burden-evidence.md |
| D11 | Reward function design | closed | Moderate | reference-configs.md, action-burden-evidence.md |
| D12 | Algorithm class | open | Moderate | reference-configs.md |
| D13 | Evaluation strategy | open | None | non-activity-interventions.md |
| D14 | Current sprint deferral | open | N/A | — |
| D15 | Feature selection (PEARL) | closed | Strong | pearl-deep-analysis.md |

---

## Conventions

- **Status** is one of: `open` / `closed` / `deferred`
- **Evidence references** follow the pattern: Author Year Journal Figure/Table — specific finding
- **Sub-questions** tagged `[parameter — sensitivity analysis]` are continuous-valued choices for sensitivity analysis
- **Deep-dive links** point to files in `state-space-design/` or `action-space-design/`
- **Upstream framing decision**: [decision-trees/online-vs-offline-rl.md](decision-trees/online-vs-offline-rl.md)

---

## Amendment log

| Date | Change |
|------|--------|
| 2026-07-01 | D3, D9, D11 closed with resolutions and key evidence; dependency diagram restructured into STATE/REWARD/ALGO bands with cross-band arrows restored |
