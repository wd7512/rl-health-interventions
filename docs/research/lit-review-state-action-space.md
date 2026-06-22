---
title: "Literature Review — State & Action Space Design for RL Health Interventions"
status: "draft v0.1"
date: "2026-06-22"
author: "W. Dennis"
purpose: "Inform state/action space extensions for feat/1a-actions, feat/1b-states, and feat/1c-reward branches"
related: "docs/initial_design.tex §3 · docs/initial_experiments/initial_experiments.tex · feat/1a-actions:docs/initial_experiments/configs/1a_actions_only.yaml · feat/1b-states:docs/initial_experiments/configs/1b_states_only.yaml"
---

# Literature Review — State & Action Space Design for RL Health Interventions

> Synthesises evidence from JITAI trials, step-count epidemiology, and
> behavioural intervention research to inform the state space, action space,
> and reward function design for the rl-health-interventions project.
> Target branches: `feat/1a-actions`, `feat/1b-states`, `feat/1c-reward`.

---

## 1. Step Count Granularity — Do People Care About High/Low/Medium or Just Thresholds?

**Short answer: thresholds win. The evidence strongly favours coarse step
categories over fine-grained counts.**

### 1.1 Dose-response evidence

| Study | N | Key finding |
|---|---|---|
| Saint-Maurice et al. (2020, JAMA) | 47,471 (15 cohorts) | Mortality risk drops steeply from ~2,700 to ~7,500 steps/day, then **plateaus at 6,000–8,000 for older adults** and ~8,000–10,000 for younger adults |
| Paluch et al. (2022, Lancet Public Health) | 15 studies, dose-response MA | 1,000-step increment → ~15% lower all-cause mortality, but curve **flattens above ~8k**; 500-step increment → 7% lower CV mortality |
| Kraus et al. (2019, Med Sci Sports Exerc) | NHANES, n=~20k | Biggest mortality jump: lowest quartile (~4,000) → Q1 (~5,500). **Diminishing returns above Q2 (~7,400)** |
| Lee et al. (2022, BJSM) | 4 studies | Minimal effective dose: **4,000–4,500 steps/day** for 50% of optimal mortality benefit |

### 1.2 The 10,000-step myth

The 10,000-step target traces to a **1965 Japanese pedometer marketing campaign**
("manpo-kei" = 10,000 steps meter). No clinical trial has ever used it as a
primary endpoint. The number persists because it's memorable, not because it's
clinically optimal.

### 1.3 Implication for state space

Don't model raw step counts as continuous states. Use **threshold-based
discrete bins** aligned with the clinical inflection points:

| Bin | Range | Rationale |
|---|---|---|
| `sedentary` | < 4,000 | Below minimum effective dose |
| `low_active` | 4,000–6,999 | Steep part of dose-response curve |
| `moderate_active` | 7,000–9,999 | Near plateau for older adults |
| `high_active` | ≥ 10,000 | Plateau region; additional benefit minimal |

The current 2-state `sedentary`/`active` collapses the entire dose-response
curve into a single threshold. **4 bins capture the clinically relevant
variation** without over-discretising.

---

## 2. Action Space Design

### 2.1 Design principles from the RL-for-health literature

**Trella et al. (2022, "Designing RL Algorithms for Digital Interventions:
Pre-Implementation Guidelines", Algorithms 15(8):255)** — the key design paper:

- Action space should be **small (2–6 actions)** to enable learning with
  limited data
- Each action should map to a **distinct behavioural construct**
- Include a **no-action/idle** option
- Actions should vary in **intensity/cost** to the user

**Liao et al. (2019, "Personalized HeartSteps", ACM IMWUT)** — the actual
HeartSteps RL system:
- Binary action: deliver vs not-deliver notification
- 5 decision points per day, ~90-day studies
- Thompson sampling with contextual features (time-of-day, location,
  weather, recent activity)

**Klasnja et al. (2019, "Efficacy of Contextually Tailored Suggestions for
Physical Activity", Annals of Behavioral Medicine)** — micro-randomized trial:
- Key finding: **timing matters as much as content** — sending suggestions
  when user is sedentary is more effective than when already active

### 2.2 Current action set (feat/1a-actions)

The `feat/1a-actions` branch defines 6 actions:

| Action | Burden | Type |
|---|---|---|
| `no_message` | 0.0 | no-op |
| `motivational_prompt` | 0.2 | activity |
| `walking_suggestion` | 0.3 | activity |
| `goal_reminder` | 0.15 | activity |
| `recovery_suggestion` | 0.25 | activity |
| `progress_feedback` | 0.1 | activity |

**Assessment**: All 6 actions are activity-focused. This misses non-physical
interventions with strong evidence (see §3 below). The burden penalties are
reasonable but arbitrary — they should be grounded in user burden literature.

### 2.3 Recommended action set (6 actions, literature-grounded)

| Action | Type | Burden | Evidence base |
|---|---|---|---|
| `idle` | no action | 0.0 | baseline |
| `gentle_nudge` | low-intensity activity prompt | 0.15 | HeartSteps "suggestion" (Liao 2019) |
| `goal_nudge` | step goal reminder | 0.1 | JustWalk protocol (JMIR Res Protoc 2023) |
| `journaling_prompt` | positive affect journaling | 0.25 | PAJ trials (Smyth 2018, JMIR Ment Health) |
| `sleep_hygiene` | sleep improvement tip | 0.1 | mHealth sleep interventions (Koffel 2018) |
| `social_encouragement` | motivational/social message | 0.2 | BCT taxonomy; HeartSteps content variants |

**Rationale for changes from current 6-action set**:
- Replaces `motivational_prompt` + `walking_suggestion` + `recovery_suggestion`
  with `gentle_nudge` + `goal_nudge` (clearer behavioural distinction)
- Adds `journaling_prompt` (see §3) — non-physical, strong evidence
- Adds `sleep_hygiene` — sleep mediates next-day activity
- Adds `social_encouragement` — distinct from informational nudges
- Removes `progress_feedback` (redundant with `goal_nudge`) and
  `no_message` (renamed to `idle`)

---

## 3. Non-Activity Actions — Journalling and Beyond

### 3.1 Expressive writing / journalling

**Smyth et al. (1999, JAMA)** — foundational Pennebaker paradigm:
- Expressive writing about stressful events improved physical health outcomes
  (fewer physician visits, improved immune function) in medical patients
- N=122, RCT, 4 sessions × 20 minutes

**Norman et al. (2004, Annals of Behavioral Medicine)**:
- Expressive writing improved health outcomes in chronic illness (asthma,
  rheumatoid arthritis)
- Objective measures: lung function (FEV1), disease activity scores

**Smyth et al. (2018, JMIR Mental Health)** — Positive Affect Journaling (PAJ):
- **Most directly relevant**: PAJ (writing about positive experiences) reduced
  anxiety/depression and improved well-being in medical patients
- N=70, RCT, 3 sessions/week × 12 weeks
- Online delivery — scalable, fits notification-based RL framework

**Baikie & Wilhelm (2005, Advances in Psychiatric Treatment)** — review:
- Expressive writing benefits: improved immune function, reduced blood
  pressure, fewer GP visits, improved lung function, improved liver function
- Mechanism: cognitive processing of emotional experiences → reduced
  physiological stress response

### 3.2 Why journalling fits the RL framework

1. **Low physical burden** (unlike exercise prompts) — can be delivered as
   a 5-minute writing exercise
2. **Measurable psychological outcomes** (mood, stress, self-efficacy)
3. **Same delivery mechanism** as activity nudges (push notification →
   in-app prompt)
4. **Different transition dynamic** — affects mood/stress states rather than
   activity states directly, creating richer state-action interactions
5. **Scalable** — no sensor required, just user self-report

### 3.3 Other non-activity actions with evidence

| Action | Evidence | Mechanism |
|---|---|---|
| Sleep hygiene prompts | Koffel et al. (2018, J Clin Sleep Med): brief sleep hygiene interventions improve sleep quality | Sleep mediates next-day activity |
| Stress management / mindfulness | Creswell et al. (2014, Psychoneuroendocrinology): brief mindfulness reduces cortisol | Stress is a barrier to physical activity |
| Self-monitoring prompts | Burke et al. (2011, J Am Diet Assoc): self-monitoring is strongest predictor of behaviour change | Increases awareness, creates feedback loop |

---

## 4. State Space Extensions

### 4.1 Current state (main branch)

`StateView(activity, day, step_of_day, steps_per_day)` — activity is binary
(`sedentary`/`active`).

### 4.2 Recommended additional dimensions

From JITAI literature (Trella 2022, Liao 2019, Klasnja 2019):

| Dimension | Values | Rationale |
|---|---|---|
| **Step bin** | `sedentary` / `low_active` / `moderate_active` / `high_active` | Dose-response inflection points (§1.3) |
| **Time-of-day** | `morning` / `midday` / `afternoon` / `evening` | HeartSteps found time-of-day is a strong contextual feature |
| **Day type** | `weekday` / `weekend` | Activity patterns differ substantially |
| **Trend** | `increasing` / `stable` / `decreasing` | Captures momentum effects (3–7 day window) |

**Optional dimensions** (require EMA or wearable data):

| Dimension | Values | Rationale |
|---|---|---|
| **Stress** | `low` / `moderate` / `high` | Key moderator of intervention effectiveness (JITAI literature) |
| **Sleep** | `poor` / `adequate` / `good` | Sleep mediates next-day activity |

### 4.3 State space size consideration

Full factorial: 4 × 4 × 2 × 3 × 3 × 3 = **864 states**. Too many for
tabular RL with limited data.

**Recommendation**: use a **factored representation** where the agent
conditions on individual features (as in contextual bandits) rather than a
flat state space. The current `contextual: true` + `context_feature` pattern
in the codebase already supports this. The agent selects actions based on
a feature vector, not a flat state index.

### 4.4 State representation in StateView

```python
@dataclass(frozen=True)
class StateView:
    activity: str          # step bin: sedentary/low_active/moderate_active/high_active
    day: int
    step_of_day: int
    steps_per_day: int
    time_of_day: str       # morning/midday/afternoon/evening
    day_type: str          # weekday/weekend
    trend: str             # increasing/stable/decreasing
    # Optional:
    # stress: str          # low/moderate/high
    # sleep: str           # poor/adequate/good
```

---

## 5. Reward Function Implications

### 5.1 Current reward

State-only reward: `sedentary=0.0`, `active=1.0`. No action cost.

### 5.2 Recommended reward structure

```
R(s, a, s') = base_reward(s') - burden_penalty(a) - fatigue_penalty(consecutive_nudges)
```

Where:
- `base_reward(s')` maps step bins to rewards (0.0 / 0.4 / 0.8 / 1.0)
- `burden_penalty(a)` from `ActionConfig.burden_penalty`
- `fatigue_penalty` increases with consecutive non-idle actions (user fatigue
  model — see Trella 2022)

### 5.3 Multi-objective considerations

The reward should reflect that different actions target different outcomes:
- Activity nudges → step count reward
- Journalling → mood/stress reward (requires EMA data)
- Sleep hygiene → sleep quality reward (requires wearable data)

For the MVP, a single scalar reward based on step bin + burden penalty is
sufficient. Multi-objective rewards are a Phase 2 extension.

---

## 6. Key References

1. Saint-Maurice PF et al. (2020). Association of daily step count and
   all-cause mortality. *JAMA*, 323(12), 1151–1160.
2. Paluch AE et al. (2022). Daily steps and all-cause mortality: a
   dose-response meta-analysis. *Lancet Public Health*, 7(3), e219–e228.
3. Trella A et al. (2022). Designing reinforcement learning algorithms for
   digital interventions: pre-implementation guidelines. *Algorithms*,
   15(8), 255.
4. Liao P et al. (2019). Personalized HeartSteps: a reinforcement learning
   algorithm for optimizing physical activity. *ACM IMWUT*, 3(1), 1–22.
5. Klasnja P et al. (2019). Efficacy of contextually tailored suggestions
   for physical activity. *Annals of Behavioral Medicine*, 53(6), 546–557.
6. Smyth JM et al. (2018). Online positive affect journaling improves
   mental distress and well-being in general medical patients.
   *JMIR Mental Health*, 5(4), e11290.
7. Smyth JM et al. (1999). Effects of writing about stressful experiences
   on symptom reduction in patients with asthma or rheumatoid arthritis.
   *JAMA*, 281(14), 1304–1309.
8. Norman SA et al. (2004). The influence of emotionally expressive writing
   on health outcomes. *Annals of Behavioral Medicine*, 27(2), 145–152.
9. Baikie KA, Wilhelm K (2005). Emotional and physical health benefits of
   expressive writing. *Advances in Psychiatric Treatment*, 11(5), 338–347.
10. Lee IM et al. (2022). Association of step volume and intensity with
    all-cause mortality in older women. *JAMA Internal Medicine*,
    179(8), 1105–1112.
