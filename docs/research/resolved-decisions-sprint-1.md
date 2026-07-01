---
title: "Resolved Decisions — Sprint 1 MDP Design"
status: "active"
date: "2026-06-28"
upstream: "decision-catalogue.md"
---

# Resolved Decisions — Sprint 1 MDP Design

> Decisions resolved during the grilling session on 2026-06-28.
> Each entry links to the relevant decision in the catalogue and
> captures the rationale, evidence basis, and open questions carried forward.

## D1. Step count encoding

**Status:** resolved — 3 step bins

3-level step bin based on per-timestep cumulative steps:
- <800 steps/timestep = inactive (<4k daily)
- 800–1,600 steps/timestep = moderate (4k–8k daily)
- >1,600 steps/timestep = active (>8k daily)

### Rationale

- Binary is too coarse — "active" masks clinical distinctions (4k vs 10k are very different health outcomes)
- 4 bins (D1 original) exceeds the minimum useful granularity; 3 bins captures the key inflection points with fewer LLM calls
- Reward is 3-level matching the bins (inactive=0.0, moderate=0.5, active=1.0), aligned with clinical thresholds
- LLM bootstrapping makes the marginal cost trivial: 3 × 4 × 3 = 36 (s, a) pairs vs 16 for binary

### Evidence

| Source | Finding |
|--------|---------|
| Lee 2022 JAMA Intern Med | Minimal effective dose ~4k-4.5k for 50% of optimal mortality benefit |
| Saint-Maurice 2020 JAMA | Mortality RR plateaus at 6k-8k (older) / 8k-10k (younger) |
| Paluch 2022 Lancet PH | 15% RR reduction per 1k steps, curve flattens above ~8k |

### Carried forward

- Engagement curve (upper bins) unknown — same as HeartSteps literature

## D2. Factored vs flat state representation

**Status:** resolved — factored with two table types

Factored into two transition structures:

**Day-boundary (step 0 — morning boundary):** `P(sleep' | step_bin, burden, action, day_of_week, sleep)`
- Sleep' is the previous night's sleep quality (observed at the start of each day)
- Only sleep changes — step_bin' is sampled next from within-day table #0 using the new sleep'

**Within-day (steps 0–4):** `P(step_bin' | step_bin, burden, action, day_of_week, sleep')`
- 5 separate tables (#0–#4), one per decision point (time-of-day is implicit as the table index)
- Only step_bin changes stochastically within a day
- Table #0 uses sleep' (just sampled from day-boundary); tables #1–#4 use the same sleep' held constant for the day

### Deterministic / formula-driven dimensions

| Dimension | Behaviour |
|---|---|
| time_of_day | Implicit — step index within day selects the within-day table (0=boundary, 1-4=within-day) |
| day_of_week | Advances deterministically, flips at 6→0 |
| burden | Rolling formula — count non-idle in last 3 timesteps |
| sleep | Transitions jointly with step_bin at day boundary only |
| goal_progress | Dropped from transition moderators |

### Rationale

- Full product of all dimensions would be intractable
- With factoring, different timescales map to different table structures naturally
- The codebase currently uses full product (`rule_based.py` reads flat state keys) and must be refactored
- Archetypes (goal-driven, social, resistant, stable maintainer) deferred from Sprint 1 — single "blank" context with no persona
- Profile variables (age, gender, baseline_activity) also deferred — not separate state dimensions

## D3. Hidden psychosocial state variables

**Status:** sleep included — mood/stress excluded

### Sleep

**2 bins:** good / poor (qualitative LLM judgment simulating smartwatch sleep-quality output).

- **Role:** A daily state dimension that transitions at the day boundary, separately from step_bin; also a reward signal (see D11)
- **Stochastic at day boundary:** yes — `P(sleep' | sleep, step_bin, burden, action, day_of_week)` is LLM-bootstrapped; step_bin' is sampled from within-day table #0 using the new sleep'
- **Static within-day:** yes — once set at step 0, sleep stays constant for all 5 within-day steps
- **Moderates within-day transitions:** yes — sleep is a context variable in the within-day LLM prompt

### Mood/stress

Excluded from Sprint 1. No deployed RL-for-health system includes them as state variables. Rabbi 2019 tested stress as moderator — null result.

### Evidence

| Source | Finding |
|--------|---------|
| Rabbi 2019 | Stress as moderator of nudge→activity: null |
| Trella 2022 | Recommends stress as optional context feature, not core |
| Irwin 2016 Lancet | Sleep interventions have moderate effect on sleep quality (g=0.43) — sufficient for inclusion as a state dimension |

### Carried forward

- Mood/stress may be revisited in Phase 2 (cross-cutting with D9)
- Sleep quality operationalisation may need sensitivity analysis vs hard threshold

## D4. Trend dimension

**Status:** excluded from Sprint 1

No trend dimension in the state.

### Rationale

- No published RL system has used a trend dimension
- No precedent, no evidence basis, purely novel/speculative
- Computation method (rolling OLS, EMA, etc.) has no literature basis

## D5. Time-of-day encoding

**Status:** resolved — implicit as table index

Time-of-day is **not a state dimension.** Instead, the step index within the day selects one of 5 within-day transition tables:

- Step 0 (morning boundary) → day-boundary table samples sleep', then within-day table #0 samples step_bin' using new sleep'
- Step 1 (morning P1) → within-day table #1 (using sleep' from step 0)
- Step 2 (P2) → within-day table #2
- Step 3 (P3) → within-day table #3
- Step 4 (evening) → within-day table #4

### Rationale

- Time-of-day is the strongest moderator in HeartSteps V2 — different times of day have different response patterns even after controlling for burden and sleep
- Making it a table index rather than a state dimension avoids multiplying the transition table while still capturing time-specific response functions
- 5 within-day tables (one per decision point, including step 0 after day-boundary) and 1 day-boundary table — the LLM bootstraps each separately
- The agent doesn't need time-of-day as a context feature to learn different policies — the different transition dynamics per time slot produce different experience distributions automatically

### Evidence

| Source | Finding |
|--------|---------|
| Liao 2019 | Time-of-day strongest moderator in HeartSteps V2 |
| HeartSteps V2 delivery | 5 decision points/day (morning + 4 within-day)

## D6. Day type encoding

**Status:** resolved — moderates transitions

Binary weekday/weekend. Day type is a **transition moderator** — it appears in both the within-day and day-boundary LLM prompts (and as a table dimension in the transition specification).

### Rationale

- Universal across all 6 reference systems
- Swapnil and Mengyan expect the design doc state space to be followed, which includes day_of_week as a context variable
- People respond differently to interventions on weekends — more free time, different routines
- The LLM bootstrap handles this with no extra hand-spec effort: day type is just another field in the prompt
- 3-level (weekday/weekend/holiday) not warranted for Phase 1

## D7. Action set composition

**Status:** resolved — 4 actions

| Action | Label | Notes |
|--------|-------|-------|
| `a_0` | idle | No intervention |
| `a_1` | movement_suggestion | Recommend a short walk or activity |
| `a_2` | goal_reminder | Remind of step goal |
| `a_3` | journal | Expressive writing / positive affect journaling |

### Rationale

- HeartSteps V2 used 2 actions (nudge, idle). We extend to 4 for richer comparison.
- idle: universal consensus, 4/6 reference systems include
- movement_suggestion: maps to HeartSteps walking suggestion — strongest evidence
- goal_reminder: distinct behavioral construct per Trella 2022 requirement
- journal: non-activity action included as placeholder/exploration; see D8
- 4 actions × 3 step bins = 12 probability rows per within-day table

### Excluded from Sprint 1

| Action | Reason |
|--------|--------|
| Motivational prompt | Overlaps with movement_suggestion — distinct construct unclear |
| Recovery/stretch | No evidence it's distinct from idle in transition effect |
| Progress feedback | Overlaps with goal_reminder — deferred |
| Social encouragement | Weakest evidence across all candidates; null in HeartSteps |

### Evidence

| Source | Finding |
|--------|---------|
| Trella 2022 | Each action should map to distinct behavioral construct, prescribes 2-6 actions |
| HeartSteps V2 | Binary actions — 34% step increase vs random |
| StepCountJITAI | 2-4 actions — burden-aware > burden-agnostic |

### Carried forward

- gentle_nudge vs goal_nudge separation has no RL comparison
- 6-action set (design doc full spec) is a natural extension sprint

## D8. Non-activity action reward

**Status:** deferred — mechanism noted for Phase 2

Journal is included in the action set but its reward mechanism is deferred. In Sprint 1, journal has no step reward benefit and carries the same burden cost as any other non-idle action. The agent will learn to avoid it — which is acceptable for the MVP.

### Mechanism noted for future

The grilling session discussed: journaling, if accepted by the user, could reduce burden. Two approaches were identified but **not resolved**:

1. **LLM-bootstrapped joint outcome:** The LLM generates `(active'_30min, journal_accepted)` jointly. Burden update depends on acceptance.
2. **Separate acceptance model:** `P(accepted | state, action)` — an additional hand-written table.

### Rationale for deferral

- Journal → burden reduction has *no empirical support* in the literature (Smyth 2018 JMIR shows journal reduces depression/anxiety, but burden reduction as an MDP mechanism is untested)
- Including it in Sprint 1 adds architectural complexity: stochastic burden, joint transition outputs, acceptance state tracking
- The existing `response_{t-1}` in the design doc's state space is designed for exactly this use case

### Evidence

| Source | Finding |
|--------|---------|
| Smyth 2018 JMIR | Journaling reduces depression/anxiety — standalone efficacy, not MDP mechanism |
| Literature review | Burden reduction via journal: **Untested** (action-burden-evidence.md) |

### Carried forward

- D9 (mood/sleep as reward vs state) is closely linked: if mood becomes a reward channel in Phase 2, journal gets a natural reward signal

## D9. Mood/sleep: reward signal vs state variable

**Status:** partially resolved — sleep is both state and reward (Sprint 1); mood-only deferred to Phase 2

Sleep is included as both a state variable (D3 — transition moderator) and a reward signal (D11 — reward component), mirroring the pattern used for `step_bin` (which also appears in both the state and reward). The two roles capture distinct constructs:

- **State role:** sleep quality moderates within-day transition dynamics — a user with poor sleep may respond differently to interventions
- **Reward role:** good sleep quality is directly valued as an outcome the agent should learn to promote

### Rationale

- Same dual-role pattern as `step_bin` — established pattern, not novel
- The two roles are conceptually distinct (moderator vs. outcome) and the LLM already distinguishes them in separate prompt types
- Mood-only component has no deployed precedent and remains deferred
- Multi-objective reward (steps + sleep) is calibrated via α weighting, not real data

### Carried forward

- Mood as reward component — deferred to Phase 2
- Trella 2022 recommends mood as optional reward component — consistent with Phase 2 timing

## D10. Burden/fatigue model

**Status:** resolved — rolling window with 3 levels

Burden is a rolling count of non-idle actions in the last 3 timesteps:

| Non-idle actions (last 3) | Burden level |
|---|---|
| 0 | low |
| 1 | medium |
| 2 or 3 | high |

### Mechanics

- The window rolls across day boundaries — the first decision point of a new day examines the last 3 steps of the previous day
- No separate decay parameter, no daily reset, no penalty values to tune
- No special-case handling needed for early timesteps (the previous day's end naturally fills the window)

### Role in the MDP

- Burden is a **table dimension** — separate LLM calls per burden level produce distinct transition distributions, no approximation
- Burden is **not** subtracted from the reward function directly (unlike StepCountJITAI's formulation). The cost of burden is expressed through reduced future activity probability.

### Rationale

- StepCountJITAI uses linear accumulator with per-action penalties and decay — but their values are heuristic
- No published system has empirically validated its burden values (action-burden-evidence.md)
- Rolling-window formulation has **zero parameters to guess** beyond the definition itself (3 timesteps, 3 levels)
- Aligns with the intuition: a user with 2-3 recent interventions is more saturated

### Evidence

| Source | Finding |
|--------|---------|
| StepCountJITAI (Karine 2024) | b(a)=0.1-0.5, d=0.2, daily reset; burden-aware > burden-agnostic |
| Trella 2022 | Burden should be reflected in reward; no specific values |
| HeartSteps V2 | No explicit burden model; frequency limited by 5 decision points/day |
| action-burden-evidence.md | **Conclusion: penalty magnitudes are universally heuristic** |

### Carried forward

- The bootstrapped per-burden-level distributions should be sanity-checked: high burden should show lower P(step_bin' > step_bin) than low burden across all actions

## D11. Reward function design

**Status:** resolved

### Final choice

```
R = α · f(step_bin') + (1-α) · g(sleep') − λ · 𝟙[action != idle]
```

Where:
- `f` maps the post-transition step bin: inactive → 0.0, moderate → 0.5, active → 1.0
- `g` maps sleep quality from the **post-transition state** `sleep'`: good → +1.0, poor → −1.0
- `α ∈ [0, 1]` weights the step and sleep reward components (default α = 0.9)
- `λ = 0.05` — a small regularisation penalty per non-idle action to discourage spamming
- Immediate per-step time horizon (30-min post-decision, matching HeartSteps V2)
- Burden is not subtracted from reward; its cost is expressed through reduced future activity probability

### Rationale

- 3-level reward matches the transition-state bins — the agent is incentivised to push users from inactive→moderate (+0.5) or moderate→active (+0.5), with no artificial threshold boundary
- λ = 0.05 is large enough to bias the agent away from spamming but small enough that it doesn't outweigh the step gain
- Burden handles the saturation dynamic; λ handles the immediate cost of interruption — two separate mechanisms
- Immediate horizon learns fastest; multi-timescale (3-week body measure) deferred to Phase 2

## D12. Algorithm class

**Status:** resolved — model-free

### Final choice

Model-free agents only. All agents learn from environment samples (the synthetic transition matrix generates experience; agents do not read it directly).

Included in Sprint 1:
- **Contextual bandits:** Thompson Sampling, Epsilon-Greedy, UCB, Decaying Epsilon-Greedy (all per ActivitySteps V2 pattern)
- **Model-free Q-learning:** optional, for non-myopic comparison

### Rationale

- Model-based methods are infeasible in clinical deployment — the ground-truth transition matrix is never available in a real trial
- HeartSteps V2 proved contextual bandits work in this domain (34% step increase)
- The synthetic transition matrix serves as the environment's simulation engine; agents experience it through sampling, not direct access
- Model-free Q-learning can be added as a comparison without changing the environment

## D13. Evaluation strategy

**Status:** resolved

### Final choice

**Baselines:**
- Random (uniform action selection)
- Idle-only (never intervene)
- Fixed-ratio (intervene every other step)

**Algorithms (compared against each other and baselines):**
- Contextual Thompson Sampling
- Contextual Epsilon-Greedy
- Contextual UCB
- Contextual Decaying Epsilon-Greedy
- Model-free Q-learning (optional)

**Metrics** (matching existing MVP tex pattern):
- Total Reward (episode sum)
- Per-Step Reward (average)
- Last 50 Steps Reward (convergence measure)
- % Gap vs Optimal Policy (where computable)

**Reporting:**
- Aggregate across all seeds
- 50 random seeds per configuration
- 450 timesteps (90 days × 5 steps/day)

### Carried forward

- Archetype evaluation (per-persona breakdown with 4 transition matrices) deferred to Phase 2
- Non-activity action evaluation (journal selection frequency ~0 with current reward) deferred to Phase 2
- Hyperparameter sensitivity analysis (epsilon, UCB c, etc.) — follow MVP tex approach

## D14. Sprint scope

**Status:** resolved — this session

### Decisions in sprint scope

| Decision | Resolution |
|---|---|
| D1 step encoding | 3 step bins (<800 / 800–1,600 / >1,600 per timestep) (resolved) |
| D2 factored/flat | Factored: within-day + day-boundary tables (resolved) |
| D3 psychosocial state | Sleep (2 bins) included; mood/stress excluded (resolved) |
| D4 trend | Excluded (resolved) |
| D5 time-of-day | Implicit — step index selects from 5 within-day tables (resolved) |
| D6 day type | Binary weekday/weekend — moderates transitions (resolved) |
| D7 action set | 4 actions: idle, movement_suggestion, goal_reminder, journal (resolved) |
| D8 non-activity reward | Deferred to Phase 2 (resolved) |
| D9 reward vs state | Partially resolved — sleep is both state and reward (dual-role, same pattern as step_bin); mood-only deferred to Phase 2 |
| D10 burden/fatigue | Rolling window, 3 levels — table dimension (resolved) |
| D11 reward design | R = α·f(step_bin') + (1-α)·g(sleep') − λ·𝟙[action≠idle]; f={inactive:0.0, moderate:0.5, active:1.0}, g={good:+1.0, poor:−1.0}, α=0.9, λ=0.05 (resolved) |
| D12 algorithm class | Model-free: contextual bandits + optional Q-learning (resolved) |
| D13 evaluation strategy | MVP metrics; per-archetype breakdown deferred to Phase 2 (resolved) |

---

## Transition matrix size summary

Six LLM-bootstrapped tables. **All dimensions are table dimensions** (separately sampled), no prompt-context approximations.

### Within-day tables (× 5, one per time step)

Five tables, each predicting step_bin' at a specific time slot. Step 0 uses the updated `sleep'` from the day-boundary table; steps 1–4 use the `sleep` that was set at the start of the day.

**All five tables have the same signature:**

```
P(step_bin' | step_bin, burden, action, day_of_week, sleep)
```

| Dimension | Bins |
|---|---|
| step_bin | 3 |
| step_bin' | 3 |
| action | 4 |
| burden | 3 |
| day_of_week | 2 |
| sleep | 2 |

**(s, a) pairs per table:** 3 × 3 × 4 × 2 × 2 = **144**
**× 5 tables:** **720 pairs total**

### Day-boundary table (× 1, step 0)

Predicts sleep' at the start of each day, based on the previous day's end-of-day state:

```
P(sleep' | step_bin, burden, action, day_of_week, sleep)
```

| Dimension | Bins |
|---|---|
| step_bin | 3 |
| sleep' | 2 |
| action | 4 |
| burden | 3 |
| day_of_week | 2 |
| sleep | 2 |

**(s, a) pairs:** 3 × 3 × 4 × 2 × 2 = **144**

### Total LLM cost

| Table | Transition inputs (dim) | Transition outputs (dim) | Cardinality (probs) | Total LLM calls | Calls per cell |
|---|---|---|---|---|---|
| Day-boundary | step_bin(3)×burden(3)×action(4)×day(2)×sleep(2) = **144** | sleep'(2) | 288 | 2,880 | 10 |
| Within-day #0 | 144 | step_bin'(3) | 432 | 4,320 | 10 |
| Within-day #1 | 144 | step_bin'(3) | 432 | 4,320 | 10 |
| Within-day #2 | 144 | step_bin'(3) | 432 | 4,320 | 10 |
| Within-day #3 | 144 | step_bin'(3) | 432 | 4,320 | 10 |
| Within-day #4 | 144 | step_bin'(3) | 432 | 4,320 | 10 |
| **Total** | **864** | | **2,448** | **24,480** | **10** |

**Calls per pair:** 20 (day-boundary, 2 outputs) or 30 (within-day, 3 outputs) — both yield exactly 10 samples per output category.

At DeepSeek V4 Flash pricing (~$0.09/M input tokens, ~$0.18/M output tokens), roughly 50–100 tokens per call → ~$0.33 total (see cost-benefit comparison below).

### State space summary

The agent observes the full state at each step:

| Dimension | Bins | Updates | Stochastic? |
|---|---|---|---|
| step_bin | 3 | Every step | Yes |
| sleep | 2 (good / poor) | Daily (step 0) | Yes (day-boundary table) |
| day_of_week | 2 | Daily (step 0) | Deterministic |
| burden | 3 | Every step | Deterministic (formula) |
| time_of_day | implicit | Not a state dimension | — |

Total state cardinality: 3 × 2 × 2 × 3 = **36 states** (time_of_day is implicit, archetype deferred).

---

## LLM bootstrapping prompt design

All 6 transition tables are bootstrapped via LLM calls through OpenRouter using **DeepSeek V4 Flash** ($0.09/M input tokens, $0.18/M output tokens).

The LLM outputs raw step counts for within-day prompts; the environment bins the result into the 3-level step_bin. The day-boundary prompt outputs a binary JSON choice.

### System prompt (cached once)

```
# Reference

You are a generally healthy adult looking to improve your exercise and sleep habits.

5 timesteps per day: morning, mid-morning, lunch, afternoon, evening

Per-timestep step ranges (daily threshold / 5):
  <800 steps     = inactive
  800-1600 steps = moderate
  >1600 steps    = active

Sleep quality: good / poor (based on how well you slept)

Burden (notification fatigue):
  low     = 0 of last 3 timesteps had an intervention
  medium  = 1 of last 3
  high    = 2 or 3 of last 3
```

### Within-day prompt (timestep 1 — morning)

```
# Current state
You just woke up. It is the morning. It is a {weekday/weekend}.
Your sleep quality was {good / poor}. Your notification fatigue is {low/medium/high}.

{Your phone just said: {movement_suggestion / goal_reminder / journal}.

How many steps do you take this timestep?
```

The LLM outputs a raw number (e.g. `800`). The environment adds it to the cumulative daily total and maps to a step bin.

### Within-day prompt (timesteps 2–5)

```
# Current state
It is the {mid-morning / lunch / afternoon / evening}. Last timestep
({morning / mid-morning / lunch / afternoon}) you took {inferred_step_count} steps.
Your notification fatigue is {low/medium/high}. It is a {weekday/weekend}.
Your sleep quality was {good / poor}.

{Your phone just said: {movement_suggestion / goal_reminder / journal}.
 -or- No action.}

How many steps do you take this timestep?
```

The `{inferred_step_count}` is the midpoint of the step_bin range from the previous timestep's output.

### Day-boundary prompt

```
# Current state
You are at the end of the day. It was a {weekday/weekend}.
Your sleep quality last night was {good / poor}. Your notification fatigue is {low/medium/high}.

Your day:
  morning:      {inferred_step_count} steps ({step_bin})
  mid-morning:  {inferred_step_count} steps ({step_bin})
  lunch:        {inferred_step_count} steps ({step_bin})
  afternoon:    {inferred_step_count} steps ({step_bin})
  evening:      {inferred_step_count} steps ({step_bin})

Your notifications today: {morning_action}, {mid-morning_action}, {lunch_action}, {afternoon_action}, {evening_action}

It's bedtime. How was your sleep quality tonight?
{"sleep_quality": "good"}
{"sleep_quality": "poor"}
```

The day-boundary prompt shows the full per-timestep breakdown from the last day — each timestep's raw step count and its corresponding bin — so the LLM can reason about how the day's activity pattern affects sleep quality.

### Cost-benefit summary

**Usage estimate (all models):**
- 24,480 LLM calls
- ~3.2M input tokens (120 avg per within-day, 180 per day-boundary)
- ~0.24M output tokens (10 avg per call)
- ~75% prompt cache hit rate (system prompt + JSON key structure cached)

**List prices (OpenRouter, June 2026):**

| Model | Input / 1M | Output / 1M | Cache read / 1M | Est. cost (no cache) | Est. cost (75% cache) |
|---|---|---|---|---|---|
| DeepSeek V4 Flash | $0.09 | $0.18 | $0.09 | **~$0.33** | **~$0.15** |
| GLM 5.2 | $0.95 | $3.00 | $0.18 | **~$3.76** | **~$1.35** |
| Claude Sonnet 4.6 | $3.00 | $15.00 | $0.30† | **~$13.20** | **~$4.60** |
| Claude Opus 4.8 | $5.00 | $25.00 | $0.50† | **~$22.00** | **~$7.40** |
| GPT-5.5 | $5.00 | $30.00 | $0.50† | **~$23.20** | **~$7.80** |

† Estimated cache read price (Anthropic/OpenAI don't publish separate cache pricing; ~10% of input price assumed).

**Prompt caching** reduces effective cost by 60–80% when the same system prompt and JSON structure is reused across calls — which is exactly our use case (identical structure, only values vary). With caching, DeepSeek V4 Flash goes from ~$0.33 to ~$0.15.

Even without caching, DeepSeek V4 Flash is **~40× cheaper** than GPT-5.5 and **~70× cheaper** output vs Claude Opus 4.8. For a bootstrapping task where any frontier model produces reasonable transition estimates, DeepSeek V4 Flash is the clear cost leader.

---

## Amendment log

| Date | Change |
|------|--------|
| 2026-06-28 | Sprint 1 config locked in at `docs/sprint1/configs/sprint1.yaml` |
| 2026-06-28 | Deferred decisions documented in `docs/research/future-sprints.md` |
| 2026-07-01 | Sleep bins → good/poor quality; sleep added to reward: R = α·f(step_bin') + (1-α)·g(sleep'); LLM prompts updated with persona background; duplicate content removed; GPT-4o mini → DeepSeek V4 Flash cost reference; decision catalogue D3/D9/D11 rationale updated; initial_state.sleep: rested → good |
