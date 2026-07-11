---
title: "Resolved Decisions — Sprint 1 MDP Design"
status: "active"
date: "2026-06-28"
upstream: "decision-catalogue.md"
---

# Resolved Decisions — Sprint 1 MDP Design

> **What this is:** Sprint 1 MDP design specification for simulated RL policy evaluation.
> Read [Algorithm 1](#algorithm-1-episode-loop) first — it defines the simulation loop
> that every other section implements.
>
> **Cost (Swapnil):** A full bootstrap costs **~$0.27** with DeepSeek V4 Flash (22,320 LLM
> calls) or **~$0.12** with prompt caching. Cheap enough to iterate — regenerate the entire
> transition matrix after any prompt tweak for pocket change.
>
> **Quick nav:** The [TOC](#decision-toc) below summarises all 13 decisions. Click any `D#` to
> jump to the full section. Inline links connect related decisions throughout.

### Algorithm 1: Episode loop

```
Algorithm 1: Run episode (N days)

Require: N days, initial state s₀ = (step_bin, burden, day_of_week, sleep)
Ensure: episode of 5N transitions

 1: daily_total ← 0
 2: state ← s₀
 3: for day = 1 to N do
 4:     for t = 0 to 4 do
 5:         a ← agent.policy(state, t)
 6:         if t = 0 then
 7:             step_bin_daily ← bin_daily(daily_total)
 8:             sleep' ← sample P_day_boundary(step_bin_daily, burden, day_of_week, sleep)
 9:             daily_total ← 0
10:         else
11:             sleep' ← state.sleep
12:         end if
13:         step_bin' ← sample P_within_day[t](step_bin, burden, a, day_of_week, sleep')
14:         daily_total ← daily_total + mid_point(step_bin')
15:         burden' ← count_non_idle_last_3(state.burden, a)
16:         day_of_week' ← advance(state.day_of_week) if t = 0 else state.day_of_week
17:         r ← α·f(step_bin') + (1-α)·g(sleep') − λ·𝟙[a ≠ idle]
18:         next_state ← (step_bin', burden', day_of_week', sleep')
19:         record (state, a, r, next_state)
20:         state ← next_state
21:     end for
22: end for
```

Where:
- `P_day_boundary` and `P_within_day[t]` are the 6 LLM-bootstrapped tables (see [Algorithm 2](#algorithm-2-bootstrap-transition-table-from-llm))
- `f(x)` = {inactive: 0.0, moderate: 0.5, active: 1.0} (see [D1](#d1-step-count-encoding))
- `g(x)` = {good: +1.0, poor: −1.0} (see [D3](#d3-hidden-psychosocial-state-variables))
- `count_non_idle_last_3(burden, a)` is a rolling 3-step window (see [D10](#d10-burdenfatigue-model))
- `bin_daily(total)` maps daily step total to per-day bin: <4000→inactive, 4000–8000→moderate, >8000→active (see [D1](#d1-step-count-encoding))
- `mid_point(bin)` maps per-timestep bin to representative steps: inactive→400, moderate→1200, active→2000

*Referenced by:* [D2](#d2-factored-vs-flat-state-representation), [D3](#d3-hidden-psychosocial-state-variables), [D5](#d5-time-of-day-encoding), [D10](#d10-burdenfatigue-model), [Transition matrix size summary](#transition-matrix-size-summary)

<a id="decision-toc"></a>
| # | Decision | Status | Summary |
|---|----------|--------|---------|
| <a id="toc-d1">D1</a> | Step count encoding | resolved | 3 bins; per-timestep (&lt;800/800–1600/&gt;1600) + daily (&lt;4k/4k–8k/&gt;8k) |
| <a id="toc-d2">D2</a> | State representation | resolved | Factored: within-day + day-boundary ([Algorithm 1](#algorithm-1-episode-loop)) |
| <a id="toc-d3">D3</a> | Psychosocial state | sleep incl. | good/poor sleep; mood/stress excluded |
| <a id="toc-d4">D4</a> | Trend dimension | excluded | No precedent, no evidence |
| <a id="toc-d5">D5</a> | Time-of-day | resolved | Implicit as table index ([Algorithm 1](#algorithm-1-episode-loop)) |
| <a id="toc-d6">D6</a> | Day type | resolved | Binary weekday/weekend moderator |
| <a id="toc-d7">D7</a> | Action set | resolved | 4 actions: idle, movement, goal, journal |
| <a id="toc-d8">D8</a> | Non-activity reward | deferred | Journal mechanism → Phase 2 |
| <a id="toc-d9">D9</a> | Mood/sleep: reward vs state | partial | Sleep dual-role resolved; mood deferred |
| <a id="toc-d10">D10</a> | Burden/fatigue | resolved | Rolling window, 3 levels ([Algorithm 1](#algorithm-1-episode-loop)) |
| <a id="toc-d11">D11</a> | Reward function | resolved | R = α·f(step_bin') + (1-α)·g(sleep') − λ·𝟙 |
| <a id="toc-d12">D12</a> | Algorithm class | resolved | Model-free: bandits (Q-learning deferred) |
| <a id="toc-d13">D13</a> | Evaluation strategy | resolved | 5 agents × 50 seeds × 450 steps |

## D1. Step count encoding

**Status:** resolved — 3 step bins

3-level step bin. Two thresholds depending on context:

| Context | Bin | Threshold | |
|---------|-----|-----------|--|
| Within-day | inactive | <800 steps/timestep | |
| | moderate | 800–1,600 steps/timestep | |
| | active | >1,600 steps/timestep | |
| Day-boundary | inactive | <4,000 daily total | |
| | moderate | 4,000–8,000 daily total | |
| | active | >8,000 daily total | |

The daily thresholds are 5× the per-timestep thresholds (5 timesteps/day). Daily bins are used only by the day-boundary transition (Algorithm 1 line 7) — the state variable itself always uses per-timestep bins.

### Rationale

- Binary is too coarse — "active" masks clinical distinctions (4k vs 10k are very different health outcomes)
- 4 bins (D1 original) exceeds the minimum useful granularity; 3 bins captures the key inflection points with fewer LLM calls
- [Reward](#d11-reward-function-design) is 3-level matching the bins (inactive=0.0, moderate=0.5, active=1.0), aligned with clinical thresholds
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

**Status:** resolved — factored with two table types ([Algorithm 1](#algorithm-1-episode-loop))

State is factored into two table structures: day-boundary (sleep' transitions) and within-day (step_bin' transitions). The per-timestep execution is defined in [Algorithm 1](#algorithm-1-episode-loop).

### Deterministic / formula-driven dimensions

| Dimension | Behaviour |
|---|---|
| time_of_day | Implicit — step index within day selects the within-day table |
| day_of_week | Advances deterministically, flips at 6→0 |
| burden | Rolling formula — count non-idle in last 3 timesteps (Algorithm 1 line 15) |
| sleep | Transitions jointly with step_bin at day boundary only |
| goal_progress | Dropped from transition moderators |

### Rationale

- Full product of all dimensions would be intractable
- With factoring, different timescales map to different table structures naturally
- The config file (`docs/sprint1/configs/sprint1.yaml`) uses a `state.factors` structure with named dimensions — separate from the existing flat `MDPConfig` model. A new `FactoredMDPConfig` pydantic model will be added alongside the old one, with `load_config()` dispatching based on top-level keys.
- Archetypes (goal-driven, social, resistant, stable maintainer) deferred from Sprint 1 — single "blank" context with no persona
- Profile variables (age, gender, baseline_activity) also deferred — not separate state dimensions

## D3. Hidden psychosocial state variables

**Status:** sleep included — mood/stress excluded

### Sleep

**2 bins:** good / poor (qualitative LLM judgment simulating smartwatch sleep-quality output).

- **Role:** A daily state dimension that transitions at the day boundary; also a reward signal (see [D11](#d11-reward-function-design))
- **Transition:** Sampled at step 0 per [Algorithm 1](#algorithm-1-episode-loop) lines 5-8 (stochastic at day boundary, static within-day)
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

- Mood/stress may be revisited in Phase 2 (cross-cutting with [D9](#d9-moodsleep-reward-signal-vs-state-variable))
- Sensitivity analysis: sleep quality vs duration threshold

## D4. Trend dimension

**Status:** excluded

No trend dimension. No published RL system has used one — no precedent, no evidence, no literature-backed computation method.

## D5. Time-of-day encoding

**Status:** resolved — implicit as table index (Algorithm 1)

Time-of-day is **not a state dimension.** The step index selects the within-day table per [Algorithm 1](#algorithm-1-episode-loop) line 3.

### Rationale

- Time-of-day is the strongest moderator in HeartSteps V2 — different times of day have different response patterns even after controlling for burden and sleep
- Making it a table index rather than a state dimension avoids multiplying the transition table while still capturing time-specific response functions
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
- journal: non-activity action included as placeholder/exploration; see [D8](#d8-non-activity-action-reward)
- 4 actions × 3 step bins = 12 probability rows per within-day table

### Excluded from Sprint 1

| Action | Reason |
|--------|--------|
| Motivational prompt | Overlaps movement_suggestion |
| Recovery/stretch | No distinct effect vs idle |
| Progress feedback | Overlaps goal_reminder |
| Social encouragement | Null result in HeartSteps |

### Evidence

| Source | Finding |
|--------|---------|
| Trella 2022 | Each action should map to distinct behavioral construct, prescribes 2-6 actions |
| HeartSteps V2 | Binary actions — 34% step increase vs random |
| StepCountJITAI | 2-4 actions — burden-aware > burden-agnostic |

### Carried forward

- gentle_nudge vs goal_nudge separation has no RL comparison (see [D10](#d10-burdenfatigue-model) for burden implications)
- 6-action set (design doc full spec) is a natural extension sprint (see [D13](#d13-evaluation-strategy) for evaluation scope)

## D8. Non-activity action reward

**Status:** deferred

Journal has no step reward benefit in Sprint 1 — the agent will learn to avoid it, which is acceptable for the MVP.

### Mechanism noted for future

Two candidate approaches identified but not resolved (see [action-burden-evidence.md](action-space-design/action-burden-evidence.md)): LLM-bootstrapped joint outcome and separate acceptance model.

### Carried forward

- [D9](#d9-moodsleep-reward-signal-vs-state-variable): if mood becomes a reward channel in Phase 2, journal gets a natural reward signal

## D9. Mood/sleep: reward signal vs state variable

**Status:** partially resolved — sleep is both state and reward (Sprint 1); mood-only deferred to Phase 2

Sleep is included as both a state variable ([D3](#d3-hidden-psychosocial-state-variables) — transition moderator) and a reward signal ([D11](#d11-reward-function-design) — reward component), mirroring the pattern used for `step_bin` (which also appears in both the state and reward). The two roles capture distinct constructs:

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

**Status:** resolved — rolling window with 3 levels (see [Algorithm 1](#algorithm-1-episode-loop) line 15)

Burden is a rolling count of non-idle actions in the last 3 timesteps:

| Non-idle actions (last 3) | Burden level |
|---|---|
| 0 | low |
| 1 | medium |
| 2 or 3 | high |

### Mechanics

Rolling window across day boundaries — no reset, no decay, no penalty values to tune. Early timesteps need no special handling (previous day's end fills the window).

### Rationale

Zero parameters to guess — burden values are universally heuristic (see action-burden-evidence.md). A user with 2-3 recent interventions is plausibly more saturated.

### Evidence

| Source | Finding |
|--------|---------|
| StepCountJITAI (Karine 2024) | b(a)=0.1-0.5, d=0.2, daily reset; burden-aware > burden-agnostic |
| Trella 2022 | Burden should be reflected in reward; no specific values |
| HeartSteps V2 | No explicit burden model; frequency limited by 5 decision points/day |
| action-burden-evidence.md | **Conclusion: penalty magnitudes are universally heuristic** |

### Carried forward

- The bootstrapped per-burden-level distributions should be sanity-checked: high burden should show lower P(step_bin' > step_bin) than low burden across all actions (see [D3](#d3-hidden-psychosocial-state-variables) for the sleep-burden interaction)

## D11. Reward function design

**Status:** resolved — config-driven formula

### Formula

```
R = alpha * step_bin_value + (1 - alpha) * sleep_value - action_penalty
```

Where:
- `step_bin_value` maps the post-transition [step bin](#d1-step-count-encoding): inactive → 0.0, moderate → 0.5, active → 1.0
- `sleep_value` maps [sleep quality](#d3-hidden-psychosocial-state-variables) from the **post-transition state** `sleep'`: good → +1.0, poor → −1.0
- `alpha` weights the step and sleep reward components (default α = 0.9)
- `action_penalty` is per-action (idle = 0.0, all others = 0.05) to discourage spamming
- Immediate per-step time horizon (30-min post-decision, matching HeartSteps V2)
- Burden is not subtracted from reward; its cost is expressed through reduced future activity probability

### Config structure

The formula is configured in `docs/sprint1/configs/sprint1.yaml`:

```yaml
reward:
  constants: {alpha: 0.9}
  variables:
    step_bin_value: {source: state.step_bin, mapping: {inactive: 0.0, moderate: 0.5, active: 1.0}}
    sleep_value:    {source: state.sleep, mapping: {good: 1.0, poor: -1.0}}
    action_penalty: {source: action, mapping: {idle: 0.0, movement_suggestion: 0.05, goal_reminder: 0.05, journal: 0.05}}
  formula: "alpha * step_bin_value + (1 - alpha) * sleep_value - action_penalty"
```

Resolved at runtime by a safe expression parser (whitelisted `+`, `-`, `*`, `/` via `ast` node-type allowlist — no `eval()`).

### Rationale

- Config-driven formula avoids hardcoding the reward structure — researchers can tweak weighting, add/remove terms, or adjust mappings in YAML
- Per-action penalty in config (not uniform) enables future differentiation between action costs without code changes
- `ast`-based parser is safe (node-type whitelist only) and zero-dependency
- λ = 0.05 is large enough to bias the agent away from spamming but small enough that it doesn't outweigh the step gain
- The sleep penalty g(poor) = −1.0 produces −0.5 reward per poor-sleep day (5 steps × 0.1 sleep weight), which is 56% of the max daily step gain (4.5) — this is intentional: carrying poor sleep forward compounds across days, giving the agent incentive to sacrifice today's steps to improve tonight's sleep
- Burden handles the saturation dynamic; λ handles the immediate cost of interruption — two separate mechanisms
- Immediate horizon learns fastest; multi-timescale (3-week body measure) deferred to Phase 2

## D12. Algorithm class

**Status:** resolved — model-free

### Final choice

Model-free agents only (see [D2](#d2-factored-vs-flat-state-representation) for the factored state representation that enables this). All agents learn from environment samples (the synthetic transition matrix generates experience; agents do not read it directly).

Included in Sprint 1:
- **Contextual bandits:** Thompson Sampling, Epsilon-Greedy, UCB, Decaying Epsilon-Greedy (all per ActivitySteps V2 pattern)
- **Context key:** All contextual agents use the full 36-state tuple `(step_bin, burden, day_of_week, sleep)` as context — every unique state is its own partition for Q-value learning
- **Model-free Q-learning:** deferred to future sprint (should be tested on the existing MVP first)

### Rationale

- Model-based methods are infeasible in clinical deployment — the ground-truth transition matrix is never available in a real trial
- HeartSteps V2 proved contextual bandits work in this domain (34% step increase)
- The synthetic transition matrix serves as the environment's simulation engine; agents experience it through sampling, not direct access
- Q-learning deferred: non-myopic comparison is interesting but adds scope; test on MVP first to validate the implementation

## D13. Evaluation strategy

**Status:** resolved

### Final choice

**Baselines:**
- Random (uniform action selection)
- Idle-only (always selects `idle`; no learning)

**Algorithms (compared against each other and baselines):**
- Contextual Thompson Sampling
- Contextual Epsilon-Greedy
- Contextual UCB
- Contextual Decaying Epsilon-Greedy

**Metrics** (matching existing MVP tex pattern):
- Total Reward (episode sum)
- Per-Step Reward (average)
- Last 50 Steps Reward (convergence measure)
- % Gap vs Optimal Policy (where computable)

**Evaluation design:**
- **One shared bootstrap** — one ~$0.12 bootstrap (Alg 2) feeds all 250 runs. Isolates policy quality, not bootstrap variance.
- **Shared seeds across agents** — 50 base seeds × 5 agent indices via `derive_agent_seed()`. Every agent sees the same initial state and transition draws. Difference = policy alone.
- 450 timesteps per run (90 days × 5 steps/day)

### Carried forward

- Per-seed bootstrapping (50 independent bootstraps to test robustness to environmental uncertainty) — Phase 2 sensitivity analysis
- Archetype evaluation (per-persona breakdown with 4 transition matrices) deferred to Phase 2 (see [D7](#d7-action-set-composition) for action set scope)
- Non-activity action evaluation (journal selection frequency ~0 with current reward) deferred to Phase 2 (see [D8](#d8-non-activity-action-reward) for journal reward discussion)
- Hyperparameter sensitivity analysis (epsilon, UCB c, etc.) — follow MVP tex approach

## Transition bootstrapping

### Algorithm 2: Bootstrap transition table from LLM

```
Algorithm 2: Bootstrap table B from LLM

Require: state dims D_state = step_bin × burden × day_of_week × sleep,
         action set A, prompt template T_B(·), samples per cell N = 10
Ensure: transition matrix P_B of size |D_state| × |A| × |D_out|

 1: for each s ∈ D_state do
 2:     for each a ∈ A do
 3:         prompt ← T_B.render(state = s, action = a)
 4:         counts ← zero(|D_out|)
 5:         for i = 1 to N do
 6:             response ← LLM.generate(prompt)
 7:             v_out ← parse(response)
 8:             counts[v_out] += 1
 9:         end for
10:         for each v_out ∈ D_out do
11:             P_B(v_out | s, a) ← counts[v_out] / N
12:         end for
13:     end for
14: end for
15: return P_B
```

Applied to 6 tables (5 within-day + 1 day-boundary). The prompt template `T_B` differs per table; see the prompt templates below.

### Table file format

Tables are serialized as **JSON** (human-readable, easy to inspect). Stored under `<repo>/tables/` for 6 files:

```
tables/
  day_boundary.json
  within_day_0.json
  within_day_1.json
  within_day_2.json
  within_day_3.json
  within_day_4.json
```

Each JSON file is a dict mapping `"{state_key}"` → `{action: {outcome: probability, ...}}`. The bootstrap script writes these; the environment's `BootstrapTransition` model loads them at init.

The `table_dir` config field in `sprint1.yaml` (`transition_model.table_dir: tables`) resolves from repo root.

*Referenced by:* Transition matrix size summary, total LLM cost, prompt design

## Transition matrix size summary

Six tables bootstrapped via [Algorithm 2](#algorithm-2-bootstrap-transition-table-from-llm). **All dimensions are table dimensions** (separately sampled), no prompt-context approximations.

### Within-day tables (× 5, one per time step)

Five tables, each predicting step_bin' at a specific time slot. Per [Algorithm 1](#algorithm-1-episode-loop), step 0 uses the updated `sleep'` from the day-boundary table; steps 1–4 use the `sleep` set at the start of the day.

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

Predicts sleep' from the previous day's total step activity (binned daily). Action does not affect sleep quality at the day boundary (Algorithm 1 line 6).

```
P(sleep' | step_bin_daily, burden, day_of_week, sleep)
```

| Dimension | Bins |
|---|---|
| step_bin_daily | 3 (<4k / 4k–8k / >8k) |
| sleep' | 2 |
| burden | 3 |
| day_of_week | 2 |
| sleep | 2 |

**(s, a) pairs:** 3 × 3 × 2 × 2 = **36** (no action dimension)

### Total LLM cost

| Table | Transition inputs (dim) | Transition outputs (dim) | Cardinality (probs) | Total LLM calls | Calls per cell |
|---|---|---|---|---|---|
| Day-boundary | step_bin_daily(3)×burden(3)×day(2)×sleep(2) = **36** | sleep'(2) | 72 | 720 | 10 |
| Within-day #0 | 144 | step_bin'(3) | 432 | 4,320 | 10 |
| Within-day #1 | 144 | step_bin'(3) | 432 | 4,320 | 10 |
| Within-day #2 | 144 | step_bin'(3) | 432 | 4,320 | 10 |
| Within-day #3 | 144 | step_bin'(3) | 432 | 4,320 | 10 |
| Within-day #4 | 144 | step_bin'(3) | 432 | 4,320 | 10 |
| **Total** | **756** | | **2,232** | **22,320** | **10** |

**Calls per (s,a) pair:** 20 (day-boundary, 2 outputs) or 30 (within-day, 3 outputs) — both yield exactly 10 samples per output category via Algorithm 2.

At DeepSeek V4 Flash pricing (~$0.09/M input tokens, ~$0.18/M output tokens), roughly 50–100 tokens per call → ~$0.27 total (see [Bootstrapping cost](#total-llm-cost)).

### State space summary

The agent observes the full state at each step (Algorithm 1):

| Dimension | Bins | Updates | Stochastic? |
|---|---|---|---|
| step_bin | 3 (per-timestep thresholds) | Every step | Yes (within-day table) |
| sleep | 2 (good / poor) | Daily (step 0) | Yes (day-boundary table) |
| day_of_week | 2 | Daily (step 0) | Deterministic |
| burden | 3 | Every step | Deterministic (formula) |
| time_of_day | implicit | Not a state dimension | — |

Total state cardinality: 3 × 2 × 2 × 3 = **36 states** (time_of_day is implicit, archetype deferred).

Note: `step_bin_daily` is not a state dimension — it is computed at the day boundary from the previous day's accumulated midpoints (Algorithm 1 line 7). The state always carries the per-timestep `step_bin`.

---

## LLM bootstrapping prompt design

All 6 transition tables are bootstrapped via [Algorithm 2](#algorithm-2-bootstrap-transition-table-from-llm) through OpenRouter using **DeepSeek V4 Flash** ($0.09/M input tokens, $0.18/M output tokens). Within-day prompts ask the LLM to output a structured JSON object `{"steps": N, "step_bin": "..."}`; the day-boundary prompt constrains output to a binary JSON choice `{"sleep_quality": "good"}` or `{"sleep_quality": "poor"}`.

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

Daily step total ranges (5 timesteps × per-timestep ranges):
  <4000 steps total     = inactive
  4000-8000 steps total = moderate
  >8000 steps total     = active

Burden (notification fatigue):
  low     = 0 of last 3 timesteps had an intervention
  medium  = 1 of last 3
  high    = 2 or 3 of last 3
```

### Within-day prompt (timestep 1 — morning)

```

# Current state
You just woke up. It is the morning. It is a {weekday/weekend}.
Last night you were {inactive / moderately active / active}.
Your sleep quality was {good / poor}. Your notification fatigue is {low/medium/high}.
{Your phone buzzes with a movement suggestion. / Your phone reminds you of your step goal. / Your phone prompts you to write in your journal.}

How many steps do you take this timestep?
Output as: {"steps": N, "step_bin": "inactive"/"moderate"/"active"}
```

The LLM outputs a structured JSON object (e.g. `{"steps": 800, "step_bin": "moderate"}`). The environment maps it to a step bin for the transition and adds `mid_point(bin)` to the day's running total. The action sentence is only rendered for non-idle actions (idle: no notification sentence).

### Within-day prompt (timesteps 2–5)

```

# Current state
It is the {mid-morning / lunch / afternoon / evening}. Last timestep
({morning / mid-morning / lunch / afternoon}) you were {inactive / moderately active / active}.
Your notification fatigue is {low/medium/high}. It is a {weekday/weekend}.
Your sleep quality was {good / poor}.

{Your phone buzzes with a movement suggestion. / Your phone reminds you of your step goal. / Your phone prompts you to write in your journal.}

How many steps do you take this timestep?
Output as: {"steps": N, "step_bin": "inactive"/"moderate"/"active"}
```

The bin label (`{inactive / moderately active / active}`) is the step bin from the previous timestep's output. The action sentence is only rendered for non-idle actions (idle: no notification sentence).
### Day-boundary prompt

```

# Current state
You are at the end of the day. It was a {weekday/weekend}.
Your sleep quality last night was {good / poor}. Your notification fatigue is {low/medium/high} (0/1/2-3 interventions in last 3 timesteps).

Today you did {daily_total} total steps ({step_bin_daily}).

It's bedtime. How was your sleep quality tonight?
{"sleep_quality": "good"}
{"sleep_quality": "poor"}
```

The prompt gives the LLM the daily step total and bin (computed by the environment from per-timestep midpoints) plus the burden level with its definition for context.

---

## Implementation summary

Build targets extracted from the Sprint 1 design decisions above. Each bullet maps to a discrete change in the codebase.

### Config & state
- New `FactoredMDPConfig` pydantic model for the `state.factors` YAML structure (alongside, not replacing, the existing `MDPConfig`)
- `load_config()` dispatches to old or new model based on top-level keys
- Factored state dataclass `(step_bin, burden, day_of_week, sleep)` replacing flat `StateView`
- State dimension metadata reads from config (dims, names, boundaries)

### Episode loop
- Python implementation of Algorithm 1: step-index loop with day-boundary vs within-day transition dispatch, burden update via rolling window formula, reward via expression parser
- `Environment.step()` adapted for factored state (accepts action, returns factored next state)

### Bootstrap transition model
- New `BootstrapTransition(TransitionModel)` — loads pre-computed JSON tables from `tables/` at init, samples next state per Algorithm 2 output
- `table_dir` config field resolves from repo root

### Bootstrap script
- Implements Algorithm 2: iterate `(s, a)` pairs per table, call DeepSeek V4 Flash via OpenRouter, parse output, normalize counts
- Writes 6 JSON files: `day_boundary.json` + `within_day_0..4.json`

### Reward expression parser
- Safe evaluator: `ast.NodeVisitor` allowlisting `Constant`, `Name`, `BinOp(+, -, *, /)`, `UnaryOp(-)`
- Reads config `reward.constants`, `reward.variables`, `reward.formula`; resolves variable values from runtime state and action
- Per-action penalty from `actions.{name}.action_penalty`

### Agents
- All 4 contextual bandits use full 36-state tuple as context key (extend `_get_context_key` to accept multi-field feature list)
- New `IdleOnlyAgent` — always selects idle, no learning
- Q-learning deferred

### Evaluation runner
- Single bootstrap → load once → all runs share the same environment dynamics
- 50 base seeds × 5 agent indices via `derive_agent_seed(base_seed, agent_index)`
- Metrics: total reward, per-step reward, last-50 convergence, % gap vs optimal
- No fixed-ratio baseline

---

## Amendment log

| Date | Change |
|------|--------|
| 2026-06-28 | Sprint 1 config locked in at `docs/sprint1/configs/sprint1.yaml` |
| 2026-06-28 | Deferred decisions documented in `docs/research/future-sprints.md` |
| 2026-07-01 | Sleep bins → good/poor; sleep added to reward; LLM prompts updated with persona; day-boundary action dimension removed; Algorithm 1 + Algorithm 2 added; duplicate content removed; D3/D9/D11 rationale updated |
| 2026-07-01 | Config-driven reward formula (`constants` + `variables` + `formula`) with safe expression parser; per-action penalty in config; Q-learning deferred to sprint after MVP; fixed-ratio dropped; idle-only added; contextual agents use full 36-state context key; JSON table format (`tables/` from repo root); `FactoredMDPConfig` model; shared bootstrap + shared seeds evaluation design |
| 2026-07-03 | Within-day prompts: bin labels replace midpoint numbers for self-narrative consistency (Jiang et al. 2024, Lu et al. 2025); natural per-action sentences with idle omission (Alomana et al. 2026). See `llm_prompting.md §Prompt-Scorecard` for rationale. |
| 2026-07-08 | Within-day prompts: structured JSON output format (`{"steps": N, "step_bin": "..."}`) replaces raw number to improve parse reliability (results.jsonl showed range responses, markdown, and verbose reasoning). See `parse.py` for parsing stubs. |
| 2026-07-10 | Morning prompt adds previous evening's step bin for table consistency (all 3 step_bin rows now produce distinct prompts). Day-boundary prompt shows burden level with definition instead of notification sequence (removes sequence sampling bias). Prompt constants extracted to `prompts/prompts.py`; `sprint1.py` contains rendering/generation only. |
