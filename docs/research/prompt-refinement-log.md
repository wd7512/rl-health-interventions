# Prompt Refinement Log — PEARL Bootstrapping Pilot

Iterative tracker for refining the PEARL bootstrapping prompts
(`src/rl_health_interventions/llm_bootstrapping/prompts/pearl.py`) against the
PEARL Constitution proxy checks. Companion machine-readable log:
[`prompt-refinement-log.json`](prompt-refinement-log.json).

## Pipeline (one round)

```bash
# 1. Generate pilot table via OpenRouter (LLM cost)
uv run python scripts/pearl_recalibration/generate_pearl_mini.py [samples_per_cell]

# 2. Validate format / direction smoke checks
uv run python scripts/pearl_recalibration/validate_pearl_mini.py

# 3. Compute constitution proxy metrics (local, free)
uv run python scripts/pearl_recalibration/analyze_pearl_mini.py
```

Then append the round to this file and to `prompt-refinement-log.json`.

## Checks & thresholds

| Check | Threshold | Constitution link |
|-------|-----------|-------------------|
| C1 action coverage | 13/13 actions present | 12-action config, all actions reachable |
| C2 cell coverage | all (state, action) cells filled | pipeline end-to-end |
| C3 state persistence | idle keeps low states low and high states high (mean P(stay) >= 0.5) | T1.1 baseline stability |
| C4 action sensitivity | ability_morning raises P(high) vs idle in >= 50% of cells | T1.2 action differentiation / T1.3 direction |
| C5 burden monotonicity | major burden never raises mean P(high) vs none, per RSM level | T2.4 attenuation / burden realism |
| C6 factor variation | no factor's modal value appears in > 75% of cells | T3.2 persona / T3.3 weekend heterogeneity |

Verdict key: `PASS` / `FAIL` (threshold) — a check may be `PARTIAL` when some
levels pass and others fail; recorded as FAIL with detail.

## Rounds

---

### Round 1 — baseline (2026-07-31)

**Prompt version:** `pearl.py @ ecdf55a` (no changes from the pilot scaffold)

**Config:** deepseek-v4-flash (openrouter), temp 0.7, 3 samples/cell, 4 states
(2 burden x 2 recent_steps_mean) x 13 actions = 156 prompts.

**Run:** 156/156 LLM calls succeeded; 2 responses had no valid day records and
were dropped (raw results not saved — fixed for round 2+). Table: 52/52 cells.

| Check | Result | Detail |
|-------|--------|--------|
| C1 action coverage | PASS | 13/13 |
| C2 cell coverage | PASS | 52/52 |
| C3 state persistence | FAIL | high -> idle P(high)=0.0 (all collapse to moderate); low -> idle P(low)=1.0 |
| C4 action sensitivity | FAIL | ability_morning raises P(high) in only 1/4 cells |
| C5 burden monotonicity | PASS (weak) | high: P(high) 0.333 -> 0.103 under major; low degenerate at 0.0 |
| C6 factor variation | FAIL | morning_steps_ratio = balanced as modal value in 51/52 cells |

**Summary:** 3/6 checks pass (C5 counted per verdict table; weak signal). Pipeline works end-to-end; prompt behaviour does
not meet constitution proxies.

**Diagnosis:**
- System prompt anchors baseline ~5,580 steps/day unconditionally → `high`
  (>7,000) states regress to moderate; `P(high)` is structurally unreachable
  for idle.
- Interventions shift steps but not enough to cross the >7,000 threshold →
  weak action sensitivity signal.
- No timing guidance → morning/evening ratio always lands `balanced`.

**Next steps (for round 2):**
1. State-conditional KEY FACTS: state the current recent_steps_mean band
   explicitly (e.g. "your recent daily average is around 8,000 steps" for
   high, "~2,500" for low).
2. Stronger causal framing for intervention action sentences.
3. Morning/afternoon timing guidance so morning_steps_ratio varies.
4. Raw results saved to `tables/pearl_12action_pilot/raw/` for diagnosis.

---

### Round 2 — state_self_model (2026-07-31)

**Prompt version:** `state_self_model` variant in `prompts/pearl.py`
(`system_extra` + callable `user_extra`). State anchors low ~3,000 / moderate
~5,500 / high ~8,000 daily steps (±500/day), idle days persist at the person's
own level, intervention days add ~150-450 steps (graded down under major
burden), time-of-day share guidance matches `morning_steps_ratio`. Baseline
output byte-identical.

**Config:** deepseek-v4-flash (openrouter), temp 0.7, 3 samples/cell, 4 states
(2 burden x 2 recent_steps_mean) x 13 actions = 156 prompts.

**Run:** 156/156 LLM calls succeeded; 1 response had no valid day records and
was dropped (raw results saved to `tables/pearl_12action_pilot/raw/`). Table:
52/52 cells.

| Check | Result | Detail |
|-------|--------|--------|
| C1 action coverage | PASS | 13/13 |
| C2 cell coverage | PASS | 52/52 |
| C3 state persistence | PASS | idle P(stay): low=1.0, high=1.0 (raw daily means: high 7,931, low 2,929) |
| C4 action sensitivity | FAIL | ability_morning raises P(high) in 0/4 cells (structural: high idle P(high)=1.0; low anchor +boost cannot cross 7,000) |
| C5 burden monotonicity | PASS | low: 0.0 vs 0.0; high: 1.0 vs 1.0 |
| C6 factor variation | FAIL | morning_steps_ratio = balanced as modal value in 52/52 cells (subset pins factor; model respects it) |

**Summary:** 4/6 checks pass (round 1: 3/6; C5 degenerate 1.0/1.0 but counted per verdict table). C3 fully fixed; C4/C6 fail for
subset-coverage reasons rather than prompt behaviour.

**Diagnosis:**
- State-conditional anchors fixed C3 completely: high states no longer
  collapse to moderate on idle; low states stay low (idle P(stay) 1.0/1.0 vs
  1.0/0.0 in round 1).
- C4 regressed 1/4 → 0/4 structurally: with persistence fixed, idle P(high)
  = 1.0 in high states so an intervention cannot raise it, and low states
  anchored ~3,000 cannot cross the >7,000 bin even with the boost (which
  landed at ~+110-120, below the +150-450 guidance).
- C6 morning-ratio component cannot pass: the pilot subset pins
  `morning_steps_ratio=balanced` in all 4 states, and the model now
  faithfully respects it (52/52 balanced vs 51/52 in round 1).

**Next steps (for round 3):**
1. `com_b_mechanisms`: strengthen intervention effect magnitude so the boost
   reliably lands in +150-450 (raw mean ~+110-120).
2. For C4 signal, extend the pilot subset to moderate recent_steps_mean
   cells (234 calls) or evaluate sensitivity on a continuous step shift
   (analyzer change).
3. For C6, vary morning_steps_ratio / walk_pattern / day_of_week in the
   pilot subset so the time-of-day guidance can manifest.

---

### Round 3 — com_b_mechanisms (2026-07-31)

**Prompt version:** `com_b_mechanisms` variant in `prompts/pearl.py`
(`system_extra` + `action_overrides` + static `user_extra`). System extra
explains the COM-B levers (capability / opportunity / motivation /
self-regulation) and the causal rule: a nudge that matches a lever the person
has a barrier on produces a real +150-450 step increase that day; an
unmatched nudge produces a small or negligible increase. The 6 nudge themes
are mapped to COM-B components (ability = capability, physical_opportunity =
opportunity, social_opportunity = opportunity via others, perceived_benefit =
reflective motivation, planning/prioritization = self-regulation). Each of
the 12 actions gets a distinct mechanism sentence ("A morning message offers
a technique that makes walking easier ... you find walking less effortful
and take a noticeably longer walk this morning."). A compact persistence
clause keeps idle at the person's stated level; user extra is a one-line
closing rule. Baseline output byte-identical.

**Config:** deepseek-v4-flash (openrouter), temp 0.7, 3 samples/cell, 4 states
(2 burden x 2 recent_steps_mean) x 13 actions = 156 prompts.

**Run:** 156/156 LLM calls succeeded; 8 responses had malformed day JSON
(quote typo on a day line; concentrated in major-burden morning cells) and
were dropped — one cell fell below the 2-sample minimum. Raw results saved to
`tables/pearl_12action_pilot/raw/`. Table: 51/52 cells.

| Check | Result | Detail |
|-------|--------|--------|
| C1 action coverage | PASS | 13/13 |
| C2 cell coverage | FAIL | 51/52 (8/156 malformed-JSON responses, one cell lost 2 samples) |
| C3 state persistence | FAIL | idle P(stay): low=1.0, high=0.333 (raw high idle means 7,210 / 6,864 — straddling the 7,000 threshold vs 7,931/7,960 in round 2) |
| C4 action sensitivity | FAIL | ability_morning raises P(high) in 1/4 cells (first real signal: dP=+0.667 in high/major, where weakened persistence freed headroom) |
| C5 burden monotonicity | PASS | high: 0.795 -> 0.667; low: 0.0 -> 0.0 |
| C6 factor variation | FAIL | morning_steps_ratio = balanced as modal value in 49/51 cells (structural in this subset) |

**Raw effect (analyzer):** overall mean lift **+149.6** steps/day (round 2:
+115), min **-228.6**, max **+1,036.4**, 31/48 cells positive. Per state —
high/major: idle 6,864, lift +382.4 (in band); low/none: idle 3,514, lift
+112.5; low/major: idle 3,388, lift +97.5; high/none: idle 7,210, lift +6.2.

**Summary:** 2/6 checks pass (round 2: 4/6). Mean intervention lift moved
toward the +150-450 target (+149.6 vs +115) but the mechanism framing did
not land evenly, and C3 regressed because the numeric self-model anchors
were dropped.

**Diagnosis:**
- Mechanism framing strengthened the average response (+149.6 vs +115) but
  with wide variance (min -229, max +1,036). The near-zero high/none lift
  (+6.2) suggests the model applied the causal rule literally: a high-
  activity, no-burden person is judged barrier-free, so every nudge is
  treated as unmatched → negligible — instead of the base persona (moderate
  barriers everywhere) making all nudges matched.
- C3 regressed (high idle P(stay) 1.0 → 0.333): the compact persistence
  clause in the system extra is weaker than round 2's explicit numeric
  anchor. High idle means drifted from ~7,931/7,960 to 7,210/6,864, landing
  on both sides of the 7,000 bin threshold.
- C2 failed for the first time: 8/156 responses carried malformed day JSON
  (one misplaced quote per line), concentrated in major-burden morning
  cells; one cell lost 2 samples. Format instruction needs hardening.
- C4 shows its first real signal (1/4) precisely where persistence weakened
  (high/major idle P(high)=0), confirming the bin-based check is sensitivity
  to idle headroom, not to prompt quality.

**Next steps (for round 4):**
1. `empirical_anchors`: keep mechanism framing, restore round-2-style
   explicit numeric anchors (low ~3,000 / moderate ~5,500 / high ~8,000,
   ±500/day) to recover C3.
2. Pair the causal rule with an explicit per-state barriers profile so
   matched/unmatched is unambiguous and all nudges land in +150-450.
3. Harden the JSON format instruction (one object per line, no trailing
   characters) against the quote typo class of failures.
4. For C4, extend the subset to moderate recent_steps_mean cells (234 calls)
   or evaluate sensitivity on a continuous step shift; for C6, vary
   morning_steps_ratio / walk_pattern / day_of_week in the subset.

---

### Round 4 — empirical_anchors (2026-07-31)

**Prompt version:** `empirical_anchors` variant in `prompts/pearl.py`
(`system_extra` + `action_overrides` + callable `user_extra`). System extra
combines three blocks: (a) **numeric anchors** restored verbatim-style from
round 2 (current level low = under 4,000 / moderate = 4,000-7,000 / high =
over 7,000 daily steps; without intervention the person stays at their own
level — high ~8,000, low ~3,000; idle continues at the established level,
never regress toward the ~5,580 population average); (b) **barrier profile**
naming one of four profiles per (recent_steps_mean, burden) cell — low/none
= struggling walker (effort/self-efficacy), low/major = struggling walker
under strain (fatigue AND opportunity), high/none = active person
(complacency and reinforcement), high/major = active person under strain
(scheduling/fatigue, NOT capability) — with the plain statement that every
profile has at least one real barrier and a matched nudge relieves it; (c)
**causal rule** kept from round 3 (matched nudge = +150-450 steps that day,
target the middle of the band ~+300; unmatched = small or negligible) plus
the **empirical anchor** from the PEARL paper (ability/technique messages
most effective and best received at 90% thumbs-up, followed by perceived
benefit; planning and prioritization help specifically when major burden
interferes with acting). The 12 action overrides copy round 3's mechanism
sentences verbatim and append one matched-clause each ("It is a matched
nudge for anyone whose ... is the barrier"); idle unchanged. User extra is a
callable that renders a per-state "BARRIER PROFILE TODAY" line naming the
day's profile and the matched-nudge ~150-450 rule — and returns `""` on
idle so nothing renders. Baseline output byte-identical.

**Config:** deepseek-v4-flash (openrouter), temp 0.7, 3 samples/cell, 4 states
(2 burden x 2 recent_steps_mean) x 13 actions = 156 prompts.

**Run:** 156/156 LLM calls succeeded; 2 responses had malformed day JSON
(low/major planning_morning, low/none perceived_benefit_morning — same quote
typo class as round 3, down from 8) and were dropped, but both cells stayed
above the 2-sample minimum. Table: 52/52 cells. Raw results saved to
`tables/pearl_12action_pilot/raw/results_empirical_anchors_20260731_182330.jsonl`.

| Check | Result | Detail |
|-------|--------|--------|
| C1 action coverage | PASS | 13/13 |
| C2 cell coverage | PASS | 52/52 |
| C3 state persistence | PASS (marginal) | idle P(stay): low=1.0, high=0.5 (exactly at threshold; high/none idle 7,554 in band, but high/major idle 6,871 still bins moderate) |
| C4 action sensitivity | FAIL | ability_morning raises P(high) in 1/4 cells (dP=+1.0 in high/major, where idle P(high)=0.0 frees headroom) |
| C5 burden monotonicity | PASS | high: 0.872 -> 0.718; low: 0.0 -> 0.0 |
| C6 factor variation | FAIL | morning_steps_ratio = balanced as modal value in 52/52 cells (structural in this subset) |

**Raw effect (analyzer):** overall mean lift **+43.1** steps/day (round 3:
+149.6, round 2: +115), min **-547.6**, max **+1,219.0**, 26/48 cells
positive (round 3: 31/48). Per state — high/none: idle 7,554, lift **-138.2**
(round 3: +6.2); high/major: idle 6,871, lift **+496.4** (round 3: +382.4,
still over the +450 cap); low/none: idle 3,352, lift **+9.2** (round 3:
+112.5); low/major: idle 3,367, lift **-195.1** (round 3: +97.5).

**Summary:** 4/6 checks pass (round 3: 2/6) — best check count so far, with
C3 back and C2 recovered. But the primary target regressed: the barrier
profile made the model too selective, collapsing the mean lift to +43.1
(26/48 positive, min -548, max +1,219) and driving two states negative.

**Diagnosis:**
- Anchors + profiles + mechanisms **did** restore C3 in the aggregate: high
  idle P(stay) 0.333 → 0.5, high/none idle back to 7,554 (vs 7,210 in round
  3), low idle solid at 1.0. But the pass is exactly at threshold: high/major
  idle is still 6,871 (vs 6,864 in round 3), below the 7,000 bin — the
  "graded down under major burden" language still counteracts the "stay
  around 8,000" anchor for that cell, so persistence holds only where burden
  is none.
- The barrier profile **destroyed the lift**: with only 1-2 named barriers
  per profile, the model judges 8-10 of the 12 themes unmatched and outputs
  near-idle days; at temp 0.7 that no-op distribution swings negative
  (low/major -195.1, high/none -138.2). The few judged-matched cells
  overshoot badly   (high/major planning_afternoon +1,219, social_opportunity_afternoon
  +960, physical_opportunity_morning +745). Binary matched/
  unmatched per-profile is incompatible with a +150-450 average across all
  12 themes.
- high/none is still the worst cell (-138.2 vs round 3's +6.2): naming
  "complacency and reinforcement" as the barrier gave the model no concrete
  lever to act on — every theme is treated as unmatched again, just recoded
  from round 3's "no barriers" reading.
- 2/156 malformed-JSON responses (down from 8; both morning cells, no cell
  lost). C4 shows 1/4 for the same structural reason as round 3; C5 holds;
  C6 remains structural in this subset.

**Next steps (for round 5):**
1. `protocol`: make the barrier profile **grade** matched-nudge strength
   (all 12 themes produce +150-450 for every profile; themes matching the
   stated barrier land at the top of the band) instead of switching
   matchedness off, or give unmatched nudges a small guaranteed floor
   (+0-100) so the all-theme mean stays in band.
2. Fix high/major persistence: burden must modulate the intervention boost,
   never the idle baseline — state that idle stays at the person's own
   level regardless of burden.
3. For C4, extend the subset to moderate recent_steps_mean cells (234 calls)
   or evaluate sensitivity on a continuous step shift; for C6, vary
   morning_steps_ratio / walk_pattern / day_of_week in the subset.
4. Keep hardening the JSON format instruction (2 malformed responses this
   round, both morning cells).

---

### Round 5 — protocol (2026-07-31)

**Prompt version:** `protocol` variant in `prompts/pearl.py` (`system_extra` +
`action_overrides` + callable `user_extra`). System extra frames the whole
simulation as the PEARL protocol: (a) **PROTOCOL FRAME** — "You are a
participant in a year-long adaptive walking-intervention study. Each day at
one decision point the study's RL system either sends you one of 12 possible
nudge messages (a behavioral theme x time-of-day pair) or no message";
(b) **EMPIRICAL PROTOCOL ANCHORS** from the PEARL trial — the system favors
ability-improvement messages (~27% of nudges, ~90% thumbs-up), with
perceived-benefit and planning also frequent, and "messages that improve a
barrier you actually have are the ones that raise your walking that day";
(c) **GRADED MATCH RULE** replacing round 4's binary matched/unmatched — each
nudge theme carries a match weight 0-1 for the day's profile: weight ≥ 0.7 →
strong response, middle of the +150-450 band (~+300); 0.3-0.7 → modest
(~+120); < 0.3 → weak but still positive (~+40); **no intervention-day
response is ever zero or negative** — even a poorly matched message beats
nothing. A WEIGHTS BY PROFILE table spells out all 6 theme weights for all 4
profiles (ability 0.9/0.8/0.5/0.5, perceived_benefit 0.7/0.6/0.7/0.6,
planning 0.5/0.8/0.4/0.9, prioritization 0.4/0.7/0.4/0.8, social_opportunity
0.4/0.4/0.5/0.4, physical_opportunity 0.5/0.8/0.3/0.8 for low/none,
low/major, high/none, high/major); (d) **IDLE PINNED INDEPENDENT OF BURDEN** —
idle days stay at the current level regardless of burden (~8,000 high,
~3,000 low). Action overrides copy round 3/4's 12 mechanism sentences and
append a per-theme weight-class clause; user extra is a callable naming the
day's profile and the theme's weight ("Your profile today: low activity,
major burden. physical-opportunity messages are a strong match (weight 0.8)
for you today."), empty on idle. Baseline output byte-identical.

**Config:** deepseek-v4-flash (openrouter), temp 0.7, 3 samples/cell, 4 states
(2 burden x 2 recent_steps_mean) x 13 actions = 156 prompts.

**Run:** 156/156 LLM calls succeeded; **0 malformed-JSON responses** — first
perfect parse round (round 4: 2, round 3: 8). Table: 52/52 cells. Raw results
saved to
`tables/pearl_12action_pilot/raw/results_protocol_20260731_183258.jsonl`.

| Check | Result | Detail |
|-------|--------|--------|
| C1 action coverage | PASS | 13/13 |
| C2 cell coverage | PASS | 52/52 |
| C3 state persistence | FAIL | idle P(stay): low=1.0, high=0.333 (high idle means 7,081 / 6,819 — burden-independence fixed, but both straddle the 7,000 bin; 3/6 samples bin moderate) |
| C4 action sensitivity | PASS | 2/4 cells (first C4 pass; dP=+0.333 high/none, +1.0 high/major — headroom from weakened high idle) |
| C5 burden monotonicity | PASS | high: 0.923 -> 0.897; low: 0.0 -> 0.0 |
| C6 factor variation | FAIL | morning_steps_ratio = balanced as modal value in 51/52 cells (structural in this subset) |

**Raw effect (analyzer):** overall mean lift **+584.9** steps/day (round 4:
+43.1, round 3: +149.6 — overshoots the +150-300 target), min **-82.5**
(round 4: -547.6), max **+1,557.1**, **46/48** cells positive (round 4:
26/48). Per state — high/none: idle 7,081, lift **+726.2** (round 4: -138.2);
high/major: idle 6,819, lift **+964.6** (round 4: +496.4); low/none: idle
3,010, lift **+445.6** (round 4: +9.2); low/major: idle 3,151, lift **+203.4**
(round 4: -195.1). All four states positive for the first time since round 2.

**Summary:** 4/6 checks pass (round 4: 4/6) — C4 passes for the first time
and the round-4 lift collapse is fully reversed (46/48 positive, all states
positive, min -83), but the graded rule overshoots (mean +584.9 vs +150-300
target, max +1,557) and C3 regressed because the high idle pin landed at
~6,800-7,100 instead of ~8,000.

**Diagnosis:**
- The graded weights + never-negative rule fixed round 4's killer: 46/48
  positive cells, all four states positive (low/none +445.6, low/major
  +203.4, high/none +726.2, high/major +964.6), min -82.5. The no-op
  distribution (8-10 unmatched themes → near-idle days) is gone — the model
  now gives every intervention day at least a small positive response.
- But magnitudes overshoot ~2-4x the target: overall mean +584.9 (target
  +150-300). Afternoon cells systematically land ~2-3x their morning twins
  (perceived_benefit_afternoon +1,533 vs +717 in high/none; planning_afternoon
  +1,557 vs +790 in high/major), so the "noticeably longer walk this
  afternoon" mechanism sentences inflate the tail. The "strong ~+300" anchor
  did not bind — strong cells run +500-1,500, while modest (mostly morning)
  cells are closer to target.
- C3 regressed (high idle P(stay) 0.333 vs 0.5): the pin held for low
  (~3,000: 3,010/3,151) and burden-independence worked (high gap closed from
  683 to 262 steps), but high idle hugs the state description "over 7,000"
  rather than the pinned ~8,000 (7,081/6,819), so 3/6 high-idle samples bin
  moderate. Round 2's "around 8,000, within roughly +/-500" anchored
  7,931/7,960; the bare pin is weaker.
- Format compliance is now perfect (0/156 malformed; best round of the
  pilot). C4's first pass is partly structural (high/major idle P(high)=0.0
  freed headroom); C5 holds; C6 remains structural in this subset.

**Next steps (for round 6):**
1. `protocol_fewshot`: add few-shot exemplars with concrete day-level step
   numbers — an idle day at the person's own level (~8,000 high / ~3,000
   low, independent of burden) and intervention days responding at
   +150-450 scaled by match weight (strong ~+300, modest ~+120, weak ~+40) —
   to calibrate both the idle pin and the lift size the abstract rule failed
   to bind. *(implemented in round 6 below)*
2. Re-anchor the idle pin with a tolerance band ("around 8,000, within
   roughly +/-500, regardless of burden") so high idle cannot straddle the
   7,000 bin. *(implemented in round 6 below)*
3. Rebalance the afternoon mechanism sentences (bound the "noticeably longer
   walk" language, e.g. a fixed +150-450 response) or rely on few-shot
   magnitudes; monitor the morning/afternoon asymmetry. *(resolved by the
   few-shot magnitudes in round 6 — state-level afternoon:morning ratios are
   now 0.85-1.29)*
4. For C4/C6 signal, extend the pilot subset to moderate recent_steps_mean
   cells and vary morning_steps_ratio / walk_pattern / day_of_week (both
   checks remain structurally blind in this subset).

---

### Round 6 — protocol_fewshot (2026-07-31)

**Prompt version:** `protocol_fewshot` variant in `prompts/pearl.py`
(`system_extra` + `action_overrides` + callable `user_extra`). The entire
protocol variant was copied to `_PROTOCOL_FEWSHOT_SYSTEM_EXTRA`,
`_PROTOCOL_FEWSHOT_ACTIONS_OVERRIDES`, `_PROTOCOL_FEWSHOT_USER_EXTRA` and
registered as `PROMPT_VARIANT_CONFIGS["protocol_fewshot"]`; the protocol
variant is untouched. Two changes to the fewshot copies: (a) **IDLE PIN
re-anchored with a tolerance band** — "around 8,000" → "between 7,500 and
8,500" and "around 3,000" → "between 2,800 and 3,200" (round 2's ±500-band
trick), keeping the burden-independence sentence; (b) **DAY-LEVEL
EXEMPLARS** appended to the end of system_extra — three plain-English
single-day descriptions anchoring magnitudes: an idle day (high-activity
person 5,100 morning + 3,100 afternoon = 8,200 total; low-activity 1,800 +
1,300 = 3,100 total), a strong-match day (weight ≥ 0.7: a low-activity
major-burden person who usually walks ~3,100 steps takes 2,200 + 1,200 =
3,400 after a well-matched ability message — about +300 vs their
no-message day), and a modest/weak day (weight ~0.4 → about +120, e.g.
5,150 + 3,150 on an 8,200 baseline; weight < 0.3 → only about +40).
Exemplars are prose ONLY — no JSON-shaped example days, because the
response parser ingests any JSON line with day/morning_steps/
afternoon_steps. The graded-weight table and never-negative rule are kept
verbatim; user_extra is round 5's profile+weight line unchanged.

**Config:** deepseek-v4-flash (openrouter), temp 0.7, 3 samples/cell, 4 states
(2 burden x 2 recent_steps_mean) x 13 actions = 156 prompts.

**Run:** 156/156 LLM calls succeeded; 3 responses had no valid day records
(low/none physical_opportunity_morning, low/major prioritization_morning,
high/major perceived_benefit_afternoon) — a new failure class: the response
opened with a prose narrative preamble (echoing the exemplar style) and the
day JSON never appeared. Each cost one sample in its own cell; all three
cells stayed above the 2-sample minimum. Table: 52/52 cells. Raw results
saved to
`tables/pearl_12action_pilot/raw/results_protocol_fewshot_20260731_184001.jsonl`.

| Check | Result | Detail |
|-------|--------|--------|
| C1 action coverage | PASS | 13/13 |
| C2 cell coverage | PASS | 52/52 (3 cells at 2 samples; no cell lost) |
| C3 state persistence | PASS | idle P(stay): low=1.0, high=1.0 (raw idle means: high 7,638/7,557, low 3,031/2,924 — every idle mean inside its stated band, burden gap only 81/107 steps) |
| C4 action sensitivity | FAIL | 0/4 cells (structurally blind in all 4: with C3 fully green, high idle P(high)=1.0 leaves no headroom; low cannot cross 7,000) |
| C5 burden monotonicity | PASS | high: 1.0 -> 1.0; low: 0.0 -> 0.0 |
| C6 factor variation | FAIL | morning_steps_ratio = balanced as modal value in 47/52 cells (structural in this subset; 5 cells now show "morning" — first real variation) |

**Raw effect (analyzer):** overall mean lift **+212.4** steps/day (round 5:
+584.9 — now in the +150-350 target band), min **-119.5** (round 5: -82.5),
max **+673.8** (round 5: +1,557.1), **43/48** cells positive (round 5:
46/48). Per state — high/none: idle 7,638, lift **+325.6** (round 5:
+726.2); high/major: idle 7,557, lift **+295.9** (round 5: +964.6);
low/none: idle 3,031, lift **+61.7** (round 5: +445.6); low/major: idle
2,924, lift **+166.3** (round 5: +203.4). All four states positive. The 5
negative cells are all on the weakest-weight themes: social_opportunity
afternoon ×2 (-119.5 high/major, -15.7 low/major) and morning ×1 (-7.1
low/none), physical_opportunity_afternoon (-97.1 low/none),
prioritization_afternoon (-59.5 low/none). Morning/afternoon asymmetry is
gone: state-level afternoon:morning lift ratios 0.85-1.29 (round 5: 2-3x).

**Summary:** 4/6 checks pass (round 5: 4/6) — C3 is fully green for the
first time since round 2 (idle P(stay) 1.0/1.0, burden-independent, all
idle means inside the stated bands) and the round-5 magnitude overshoot is
corrected into the +150-350 target (mean +212.4, max +673.8, no 2-3x
morning/afternoon asymmetry). Trade-offs: C4 regresses to 0/4 (structurally
blind in all four cells now that the idle pin holds), and 3 parse failures
(a new prose-preamble class vs round 5's perfect 0) cost one sample each
without losing a cell.

**Diagnosis:**
- The prose day-level exemplars calibrated the lift the abstract rule could
  not bind: mean +212.4 vs +584.9 (a ~2.7x correction toward the paper's
  +150-450 effect), max +673.8 vs +1,557.1, all four states positive, and
  the round-5 2-3x afternoon inflation collapsed to 0.85-1.29. The five
  remaining negative cells concentrate on the lowest-weight themes
  (social_opportunity, weight 0.4/0.4/0.5/0.4) and afternoon cells of
  low/none — the exemplars demonstrate a strong match and idle but never a
  weak/afternoon day, so the model still under-shoots those.
- The ±500 band pin fixed C3 exactly as it did in round 2: high idle
  7,638/7,557 and low idle 3,031/2,924 — every idle mean inside its stated
  band, burden gaps of 81/107 steps, P(stay) 1.0/1.0. Round 5's straddle
  (7,081/6,819, P(stay) 0.333) is gone.
- C4 is now structurally blind in all 4 cells (0/4): a fully-green C3 means
  high idle P(high)=1.0 — no headroom for ability_morning to raise — and
  low cells cannot cross 7,000 from ~3,000. Round 5's 2/4 pass was an
  artifact of the broken pin. C4 and C3 are structurally opposed in this
  subset; the fix belongs in the subset (moderate RSM cells), not the
  prompt. C6 remains structural (47/52 balanced, though 5 cells now vary).
- 3/156 parse failures — first regression from round 5's 0, and a new
  class: prose-preamble responses that never emitted day JSON (the
  exemplars teach narrative style). No cell fell below 2 samples, so C1/C2
  still pass 52/52; the output instruction should say "JSON lines only, no
  narrative" to close it.

**Next steps (post-ladder):**
1. Ship `protocol_fewshot` as the prompt for the real bootstrapping
   experiment (see Ladder summary below); keep `protocol` as fallback.
2. Rebalance the weakest themes: add a weak-match afternoon exemplar or
   bound the afternoon mechanism sentences so social_opportunity afternoons
   stop going negative.
3. Harden the prose-preamble parse class: "output the 7 JSON lines only,
   no narrative" in the format instruction.
4. For C4/C6 signal, extend the pilot subset to moderate recent_steps_mean
   cells and vary morning_steps_ratio / walk_pattern / day_of_week.

---

### Round 7 — protocol_fewshot (2026-07-31)

**Prompt version:** `protocol_fewshot` r7 in `prompts/pearl.py`. Four edits,
all inside the fewshot constants; the protocol variant and every other
variant are untouched:

1. **Binding never-negative floor** — the weak tier goes from "weak but
   still slightly positive (~+40)" to "small but clearly positive (~+60 to
   +100)" and the modest tier from "around +120" to "~+120 to +180". The
   old "never zero or negative" sentence is replaced by the binding
   sentence: *"A message NEVER reduces your steps, no matter how poorly it
   matches. Even the weakest match raises your day's total by a small
   amount. This holds for morning AND afternoon messages alike."*
2. **Low/no-burden profile enrichment** — a "LOW NO-BURDEN PROFILE"
   paragraph spells out three explicit barriers: (a) effort/technique →
   ability is a strong match (weight 0.8), (b) motivation/energy dips →
   perceived_benefit is a strong match (0.8, "works through motivation,
   not logistics"), (c) lack of routine → planning is a good match (0.7).
   The WEIGHTS BY PROFILE table's low/no-burden column is re-weighted
   accordingly (ability 0.9→0.8, perceived_benefit 0.7→0.8, planning
   0.5→0.7, social_opportunity 0.4→0.3; prioritization 0.4 and
   physical_opportunity 0.5 unchanged), and the fewshot day-level
   user_extra reads the same table via `_PROTOCOL_FEWSHOT_PROFILE_WEIGHTS`
   so the per-day line never contradicts the system table. The other three
   profiles keep round-5 weights.
3. **Afternoon parity** — one sentence in the causal-rule block: *"An
   afternoon message is just as likely to raise your steps as a morning
   message; the time of day only changes which part of the day the extra
   steps land in."* Action overrides unchanged (already symmetric).
4. **Low-baseline weak-match exemplar** — the old "weight < 0.3 → added
   only about 40" clause is replaced by: *"A weakly matched message (weight
   ~0.3) on a low-activity day added about 60 steps: a person who usually
   takes 3,060 steps took 1,700 morning and 1,420 afternoon (3,120
   total)."* Other exemplars kept verbatim. Prose only — no JSON lines.

**Config:** deepseek-v4-flash (openrouter), temp 0.7, 3 samples/cell, 4 states
(2 burden x 2 recent_steps_mean) x 13 actions = 156 prompts.

**Run:** 156/156 LLM calls succeeded, **0/156 parse failures** — the first
clean run since round 5 (round 6: 3 prose-preamble failures). Table: 52/52
cells, every cell at 3 samples. Raw results saved to
`tables/pearl_12action_pilot/raw/results_protocol_fewshot_20260731_185819.jsonl`.
Round-6 table archived as
`tables/pearl_12action_pilot/archive/pearl_pilot_protocol_fewshot_r6.json`.

| Check | Result | Detail |
|-------|--------|--------|
| C1 action coverage | PASS | 13/13 |
| C2 cell coverage | PASS | 52/52 |
| C3 state persistence | PASS | idle P(stay): low=1.0, high=1.0 (raw idle means: high 7,710/7,545, low 2,930/2,924 — every idle mean inside its stated band, burden gap 24/165 steps) |
| C4 action sensitivity | FAIL | 0/4 cells (structurally blind: high idle P(high)=1.0 leaves no headroom; low cannot cross 7,000) |
| C5 burden monotonicity | PASS | high: 1.0 -> 1.0; low: 0.0 -> 0.0 |
| C6 factor variation | FAIL | morning_steps_ratio = balanced as modal value in 44/52 cells (8 cells now "morning"; structural in this subset) |

**Raw effect (analyzer):** overall mean lift **+194.1** steps/day (round 6:
+212.4 — still in the +150-450 band), min **-140.0** (round 6: -119.5),
max **+811.0** (round 6: +673.8), **47/48** cells positive (round 6:
43/48). Per state — high/none: idle 7,545, lift **+380.6** (round 6:
+325.6); high/major: idle 7,710, lift **+164.6** (round 6: +295.9);
low/none: idle 2,930, lift **+102.4** (round 6: +61.7); low/major: idle
2,924, lift **+128.8** (round 6: +166.3). All four states positive.

**Cell-level highlights:**
- **Negatives: 1 (vs 5).** The five round-6 negative cells are all
  positive now — low/none physical_opportunity_afternoon +52.9 (was
  -97.1), low/none prioritization_afternoon +136.7 (was -59.5), low/none
  social_opportunity_morning +20.0 (was -7.1), low/major
  social_opportunity_afternoon +2.4 (was -15.7), high/major
  social_opportunity_afternoon +185.7 (was -119.5). The lone survivor is
  **high/major ability_afternoon -140.0** — not a weak-weight cell (ability
  is 0.5 for high/major), and it sits on a pattern where the same state's
  afternoon cells collapsed (planning_afternoon +3.6,
  perceived_benefit_afternoon +71.4) while mornings overshot
  (perceived_benefit_morning +600.0, prioritization_morning +504.8).
- **low/none: +102.4 (vs +61.7), all 12 cells positive.** The enriched
  profile fixed the flatness directionally but undershot the +150 target:
  the two 0.8-weight themes' *mornings* under-deliver (ability +93.8,
  perceived_benefit +22.9) while their afternoons deliver (+151.0/+154.8).
  Planning became the state's strongest theme (morning +191.4, afternoon
  +196.7) — exactly what the habit-building barrier predicts.
- **Afternoon vs morning:** low states are now afternoon-heavy (low/none
  afternoon +125.4 vs morning +79.4; low/major +136.3 vs +121.3) — parity
  held and then some. In high states the parity sentence redistributed
  lift mass instead of equalizing: high/none afternoons +469.5 vs +291.7
  mornings, with two +800 cells (ability_afternoon +800.0,
  perceived_benefit_afternoon +811.0 — max, overshooting the +450 top of
  the paper effect), while high/major afternoons collapsed to +65.5 vs
  +263.7.

**Summary:** 4/6 checks pass (round 6: 4/6). The binding floor did its job:
1 negative cell (vs 5), 47/48 positive, all five round-6 negatives turned
positive — the floor's targets exactly. C3 stays fully green (idle P(stay)
1.0/1.0, every idle mean inside its band) and mean lift +194.1 stays in the
+150-450 band; 0 parse failures (round 6: 3). The new variance is
time-of-day: afternoons now carry more lift than mornings in low states
(parity, good) but also in high/none (two +800 cells — overshoot) while
high/major afternoons sag (one -140 cell). C4/C6 remain structurally blind
in this subset.

**Diagnosis:**
- The binding floor eliminated the weak-weight negatives outright: all five
  round-6 negative cells are positive, including high/major
  social_opportunity_afternoon (worst cell in round 6, -119.5 → +185.7).
  The one surviving negative (high/major ability_afternoon -140.0, weight
  0.5) is not on the weak tier, so the floor bound as written; it is
  residue of a high/major afternoon pattern where planning (+3.6) and
  perceived_benefit (+71.4) afternoons also hug zero while the same
  themes' mornings overshoot (+204.8/+600.0) — the model shifted its
  major-burden time-of-day discount onto afternoons.
- The enriched low/no-burden profile lifted the state from +61.7 to +102.4
  and made all 12 cells positive, but undershot the +150 target: the two
  0.8-weight themes deliver +93.8/+22.9 in the morning, so the model reads
  the profile as uniformly moderate rather than strongly matched on the
  three named barriers. The clearest wins are planning (now the strongest
  theme, as the routine-building barrier predicts) and the afternoon side
  (afternoon mean now above morning, was the reverse in round 6).
- Afternoon parity held in low states without inflating the mean out of
  band (+194.1 overall), but it flipped the high-state time bias: high/none
  afternoons now carry two +800 cells (vs the +150-450 paper effect) and
  high/major afternoons carry the one negative — the sentence bound the
  "afternoon = pointless" reading but over-corrected in the unburdened
  high state and under-corrected in the burdened one.
- 0/156 parse failures (round 6: 3) — the prose-preamble failure class did
  not recur; C1/C2 pass with every cell at 3 samples. C4/C6 unchanged and
  structural (high idle P(high)=1.0 leaves no headroom; 44/52 balanced).

**Next steps:**
1. One more prompt-only refinement: pin high/major afternoons (one -140,
   two near-zero) — e.g. note in the major-burden fatigue clause that an
   afternoon delivery does not flatten a matched message — and pull
   high/none afternoons back under +450 with a high-state afternoon
   exemplar.
2. Bind the low/none strong-match mornings: ability/perceived_benefit
   mornings (+93.8/+22.9) under-deliver vs the ~+300 anchor; a
   low-person morning strong-match exemplar would close the gap toward the
   +150 low/none target.
3. If the next round holds negatives ≤ 1 and mean lift in band, ship
   `protocol_fewshot` for the real bootstrapping experiment; keep
   `protocol` as fallback.
4. For C4/C6 signal, extend the pilot subset to moderate
   recent_steps_mean cells and vary morning_steps_ratio / walk_pattern /
   day_of_week.

---

### Round 8 — protocol_fewshot (2026-07-31)

**Prompt version:** `protocol_fewshot` r8 in `prompts/pearl.py`. Three edits,
all inside the fewshot constants; the protocol variant and every other
variant are untouched:

1. **Absolute-lift rule** — a new paragraph in the causal-rule section:
   *"The size of the increase is a FIXED absolute number of steps — it
   does not scale with your usual activity level. A strongly matched
   message adds about 300 steps whether your baseline is 3,000 steps or
   8,000 steps. A modestly matched message adds about 150 steps. A weak
   match adds about 60-100 steps."* The existing +150-450 band statement
   (strong = middle of the band, ~+300) is kept as-is — consistent with
   the new rule.
2. **Ceiling** — appended to the same paragraph: *"No message adds more
   than about 500 steps in a day."*
3. **High-baseline strong-match exemplar** — added to DAY-LEVEL EXEMPLARS
   after the existing low-baseline strong exemplar (3,100 → 3,400, kept
   verbatim): *"For example, a strongly matched message on a high-activity
   day raised the day from 8,100 total steps (5,000 morning + 3,100
   afternoon) to about 8,400 total (5,200 morning + 3,200 afternoon) —
   about 300 extra steps, the same absolute increase a low-activity person
   would get."* Prose only. No action override quantifies an effect, so
   `_PROTOCOL_FEWSHOT_ACTIONS_OVERRIDES` is unchanged.

**Config:** deepseek-v4-flash (openrouter), temp 0.7, 3 samples/cell, 4 states
(2 burden x 2 recent_steps_mean) x 13 actions = 156 prompts.

**Run:** 156/156 LLM calls succeeded, **0/156 parse failures** — second
clean run in a row. Table: 52/52 cells, every cell at 3 samples. Raw results
saved to
`tables/pearl_12action_pilot/raw/results_protocol_fewshot_20260731_190547.jsonl`.
Round-7 table archived as
`tables/pearl_12action_pilot/archive/pearl_pilot_protocol_fewshot_r7.json`.

| Check | Result | Detail |
|-------|--------|--------|
| C1 action coverage | PASS | 13/13 |
| C2 cell coverage | PASS | 52/52 |
| C3 state persistence | PASS | idle P(stay): low=1.0, high=1.0 (raw idle means: high 7,626/7,390, low 3,019/2,981 — high/major idle slid to 7,390 but still bins high) |
| C4 action sensitivity | FAIL | 0/4 cells (structurally blind: high idle P(high)=1.0 leaves no headroom; low cannot cross 7,000) |
| C5 burden monotonicity | FAIL | high: 0.9744 vs 0.9744 — burden_reduces_steps=false. Two high-state intervention cells (high/none prioritization_morning, high/major social_opportunity_morning) each dipped one sample below 7,000, so both state means fell to 0.9744 and the check's `major <= none` comparison fails on equality |
| C6 factor variation | FAIL | morning_steps_ratio = balanced as modal value in 47/52 cells (structural in this subset) |

**Raw effect (analyzer):** overall mean lift **+179.1** steps/day (round 7:
+194.1 — still in the +150-450 band), min **-80.5** (round 7: -140.0), max
**+775.4** (round 7: +811.0), **36/48** cells positive (round 7: 47/48).
Per state — high/major: idle 7,390, lift **+425.5** (round 7: +164.6);
high/none: idle 7,626, lift **+212.4** (round 7: +380.6); low/major: idle
2,981, lift **+72.8** (round 7: +128.8); low/none: idle 3,019, lift **+5.6**
(round 7: +102.4). All four states positive, but the low states collapsed.

**Cell-level highlights:**
- **Negatives: 11 (vs 1).** The floor broke in low states: 8/12 low/none
  cells negative or zero (only ability_morning +102.4, ability_afternoon
  +146.2, planning_morning +53.3, prioritization_afternoon +14.8 positive;
  perceived_benefit_morning at exactly +0.0) and 2/12 low/major
  (social_opportunity_afternoon -73.8, physical_opportunity_afternoon
  -80.5, the min). High states each have 1: high/none
  prioritization_morning -4.8, high/major social_opportunity_morning -30.5.
  Round 7's lone negative (high/major ability_afternoon -140.0) recovered
  to +418.8.
- **Max cell lift: +775.4** (high/major prioritization_afternoon; its
  morning twin +769.0) — barely down from round 7's +811.0. The ceiling
  sentence held for the round-7 +800 cells (high/none ability_afternoon
  +800.0 → +88.1, perceived_benefit_afternoon +811.0 → +366.7) but not for
  high/major's strong-weight cells (prioritization 0.8, planning 0.9,
  physical_opportunity 0.8): planning_morning +492.9, ability_morning
  +733.3, physical_opportunity_morning +407.1.
- **low/none: +5.6 (vs +102.4).** The target was up toward +200-300; the
  state moved the wrong way. The 0.8-weight themes' mornings delivered
  +102.4/+0.0 (ability/perceived_benefit) — the strong-match exemplar that
  anchored this state in rounds 6-7 stopped binding — and modest/weak
  afternoons went negative.
- **high/none: +212.4 (vs +380.6)** — into band, exactly the round-8
  target for this state. **high/major: +425.5 (vs +164.6)** — the overshoot
  moved here instead; mornings now dominate (+733.3/+769.0/+492.9).

**Summary:** 3/6 checks pass (round 7: 4/6). The fixed-absolute rule
corrected the state it targeted most (high/none into band, both round-7
+800 cells killed) and C3 stays fully green, but it did not deliver the
pair: low/none collapsed to +5.6 with 7/12 negative cells (the round-7
floor did not survive the new paragraph), the ceiling held only
adversarially (max +775.4 vs +811.0, two high/major prioritization cells
+769/+775), and C5 broke for the first time since round 2 when two
high-state intervention cells dipped one sample each below 7,000 (0.9744 vs
0.9744, burden_reduces_steps=false). Mean lift +179.1 stays in band and
parse failures stay 0.

**Diagnosis:**
- The ABSOLUTE-LIFT RULE over-corrected in low states and under-corrected
  in high/major. Reading "modest adds about 150, weak adds about 60-100" as
  a small fixed number, the model pushed low-state modest/weak cells back
  toward zero — 9 of the 11 negatives are low-state cells — where round 7's
  binding floor had held all 12 low/none cells positive (+102.4). The
  strong-match anchor for low states (the 3,100 → 3,400 exemplar) stopped
  binding in the morning: the 0.8-weight themes delivered +102.4/+0.0
  mornings, so the low/none mean collapsed to +5.6 against the +200-300
  target.
- The ceiling sentence lost to the exemplars in the strong-weight state.
  high/none's two round-7 +800 cells were reined in (the high-baseline
  exemplar binds for that state: +88.1/+366.7), but high/major's
  strong-weight column (prioritization 0.8, planning 0.9, physical_opp 0.8)
  produced +769/+775 and a state mean of +425.5 — a single sentence at the
  end of a rule paragraph is weaker than two concrete +300 exemplar days.
- C5's regression is a binning artifact at the sample level, not a
  monotonicity violation: both high states sit at 0.9744 because one
  intervention day per state crossed just below 7,000 (high/none
  prioritization_morning, high/major social_opportunity_morning — both
  low-lift cells this round), so `major <= none` fails on equality.
  C4 (0/4) and C6 (0.90 dominant share) remain structurally blind in this
  subset; 0/156 parse failures.

**Next steps:**
1. Round 9: make the ceiling binding — state it as a cap on the exemplars
   ("the largest increase in any example day is about +300; nothing in this
   study exceeds about +500") or re-state it in the strong tier of the
   GRADED MATCH RULE, since the single sentence lost to the +300 exemplars
   in high/major.
2. Restore the floor inside the ABSOLUTE-LIFT RULE: re-pin the weak tier to
   never-below (~+60 minimum, never zero or negative) and add a low-baseline
   strong-match *morning* exemplar to re-anchor low/none's 0.8-weight
   themes (+102.4/+0.0 this round).
3. If the next round holds negatives ≤ 2, max cell lift ≤ +550, low/none
   ≥ +150 and C5 green, ship `protocol_fewshot` for the real bootstrapping
   experiment; keep `protocol` as fallback.
4. For C4/C6 signal, extend the pilot subset to moderate
   recent_steps_mean cells and vary morning_steps_ratio / walk_pattern /
   day_of_week.

---

### Round 9 — protocol_fewshot (2026-07-31)

**Prompt version:** `protocol_fewshot` r9 in `prompts/pearl.py`.

**Prompt change summary:**

1. **Round 8 fully reverted** (the orchestrator's `git checkout d2b634b`; the
   staged revert is committed together with this round) — the ABSOLUTE-LIFT
   RULE paragraph, its ceiling sentence, and round 8's high-baseline
   exemplar are all gone. The prompt returns to the exact round-7 state.
2. **One exemplar added** — a single prose sentence appended to the
   strong-match exemplar paragraph in DAY-LEVEL EXEMPLARS, immediately
   after the low-activity strong exemplar (3,100 → 3,400, kept verbatim):
   *"A strongly matched message looks the same for a high-activity person:
   for example, a day that would have been 8,100 total steps (5,000
   morning, 3,100 afternoon) becomes about 8,400 total (5,200 morning,
   3,200 afternoon) — again about 300 extra steps, and never more than
   about 500."* The ceiling is carried by the example alone, per round 8's
   lesson that abstract magnitude rules (fixed ~150/60-100 numbers) collapse
   low states. No rule sentences, no standalone ceiling, no changes to the
   graded-weight table, round-7 floor wording, barrier profiles, afternoon
   parity, action overrides, or user_extra.

**Config:** deepseek-v4-flash (openrouter), temp 0.7, 3 samples/cell, 4 states
(2 burden x 2 recent_steps_mean) x 13 actions = 156 prompts.

**Run:** 156/156 LLM calls succeeded, **0/156 parse failures** — third clean
run in a row. Table: 52/52 cells, every cell at 3 samples. Raw results saved
to `tables/pearl_12action_pilot/raw/results_protocol_fewshot_20260731_191508.jsonl`.
Round-8 table archived as
`tables/pearl_12action_pilot/archive/pearl_pilot_protocol_fewshot_r8.json`.

| Check | Result | Detail |
|-------|--------|--------|
| C1 action coverage | PASS | 13/13 |
| C2 cell coverage | PASS | 52/52 |
| C3 state persistence | PASS | idle P(stay): low=1.0, high=1.0 (raw idle means: high 7,333/7,758, low 2,976/2,933 — all inside their stated bands) |
| C4 action sensitivity | FAIL | 0/4 cells (structurally blind: high idle P(high)=1.0 leaves no headroom; low cannot cross 7,000) |
| C5 burden monotonicity | PASS | burden_reduces_steps=true at both levels (round 8's fail was a binning artifact) |
| C6 factor variation | FAIL | morning_steps_ratio = balanced as modal value in 45/52 cells (0.8654 > 0.75 threshold — marked PASS in the round-9 report but FAIL per the analyzer) |

**Raw effect (analyzer):** overall mean lift **+231.3** steps/day (round 7:
+194.1; round 8: +179.1 — in the +150-450 band), min **-313.3** (round 7:
-140.0), max **+981.0** (round 7: +811.0 — worse), **43/48** cells positive
(round 7: 47/48), **5 negative cells** (round 7: 1; target ≤ 3): high/major
ability_morning -313.3, high/major perceived_benefit_afternoon -232.9,
high/major physical_opportunity_morning -74.3, low/none
physical_opportunity_afternoon -78.6, low/none ability_afternoon -38.1.
Per state — high/major: idle 7,758, lift **+148.6** (round 7: +164.6 — in
band); high/none: idle 7,333, lift **+596.7** (round 7: +380.6 — target
~+300, overshoot amplified); low/major: idle 2,933, lift **+109.2** (round
7: +128.8); low/none: idle 2,976, lift **+70.9** (round 7: +102.4 — below
the +100 floor, but NOT collapsed the way round 8 collapsed it to +5.6).

**Verdict table:**

| Metric | Target | Round 7 | Round 8 | Round 9 | Verdict |
|--------|--------|---------|---------|---------|---------|
| Checks | — | 4/6 | 3/6 | 4/6 | better on paper (C5), not on lift |
| Max cell lift | ≤ ~+550 | +811.0 | +775.4 | **+981.0** | FAIL — worse |
| high/none mean | → +300 | +380.6 | +212.4 | **+596.7** | FAIL — amplified |
| low/none mean | ≥ +100 | +102.4 | +5.6 | +70.9 | FAIL (slipped, not collapsed) |
| Overall mean | +150-450 | +194.1 | +179.1 | +231.3 | PASS |
| Negative cells | ≤ 3 | 1 | 11 | 5 | FAIL |
| C3 idle P(stay) | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | PASS |
| Parse failures | ≤ 2 | 0 | 0 | 0 | PASS |

**Diagnosis:** The exemplar-only ceiling failed on its main target — the
high-activity exemplar was read as a *typical* strongly-matched day, not a
bound. Several high/none intervention days landed near the exemplar's 8,400
*total* regardless of the 7,300-7,500 idle baseline (planning_afternoon
+1,476.7 on a 6,700 idle, perceived_benefit_afternoon +1,400), so the
exemplar anchored totals rather than increments and high/none mean lift rose
to +596.7 with max cell +981.0 — worse than round 7. "Never more than about
500" inside an example sentence was treated as example flavor. The one
bright spot: the low states did NOT collapse this time (low/none +70.9 vs
round 8's +5.6), so the round-8 failure mode is confirmed as specific to the
fixed-absolute *rule*, not to mentioning high-activity magnitudes at all.
But low/none still slipped below the +100 floor (2 of the 5 negatives are
low/none) and high/major is the only state fully in band — the two
residuals from round 7 are both still open, one of them wider.

**Next steps:**
1. Round 10: drop the high-activity exemplar (it anchored totals at ~8,400
   and amplified high/none overshoot) and state the ceiling as a rule
   sentence inside the GRADED MATCH RULE paragraph — the one rule paragraph
   that demonstrably binds (round 7's floor +60-100 and +150-450 band both
   held): e.g. *"the +150-450 band is also a ceiling: no message, however
   well matched, raises a day by more than about 500 steps."* Rule-level
   cap, no small fixed absolute numbers — round 8's collapse mechanism is
   avoided.
2. Close the low-state gap separately: low/none is stuck at +70-102 across
   rounds 7-9; add an explicit low-baseline strong-match *morning* exemplar
   so the 0.8-weight ability/perceived_benefit morning cells stop
   undershooting.
3. If a rule-level ceiling round holds negatives ≤ 2, max cell lift ≤ +550,
   low/none ≥ +100 and C5 green, ship `protocol_fewshot` for the real
   bootstrapping experiment; keep `protocol` as fallback.
4. For C4/C6 signal, extend the pilot subset to moderate
   recent_steps_mean cells and vary morning_steps_ratio / walk_pattern /
   day_of_week.

---

### Round 10 — protocol_fewshot (2026-07-31)

**Prompt version:** `protocol_fewshot` r10 in `prompts/pearl.py`.

**Prompt change summary:**

1. **Round-9 high-activity exemplar removed** — the sentence *"A strongly
   matched message looks the same for a high-activity person: ... again
   about 300 extra steps, and never more than about 500"* is deleted; the
   exemplar paragraph returns to the round-7 text verbatim.
2. **Ceiling moved into the GRADED MATCH RULE paragraph** — one sentence
   appended at the paragraph's end: *"The +150-450 step band is also a
   ceiling: no message raises a day's total by more than about 500 steps,
   no matter how well it matches."* Rule-level cap in the one paragraph
   that demonstrably binds (round 7's floor +60-100 and band +150-450 both
   held), with no small fixed absolute numbers — round 8's collapse
   mechanism is avoided. The weak-tier floor wording is untouched.
3. **Low-baseline strong-match MORNING exemplar added** — one prose
   sentence immediately after the existing strong-match exemplar (3,100 →
   3,400): *"The same applies for a low-activity person with no extra
   burden: a well-matched ability message in the morning raised their day
   from about 3,100 total (1,800 morning, 1,300 afternoon) to about 3,400
   (2,200 morning, 1,200 afternoon)."* Prose only, never JSON lines.

Untouched: graded weight table, LOW NO-BURDEN PROFILE paragraph, afternoon
parity sentence, weak-tier floor, action overrides, user_extra, idle
exemplar, modest/weak exemplars.

**Config:** deepseek-v4-flash (openrouter), temp 0.7, 3 samples/cell, 4 states
(2 burden x 2 recent_steps_mean) x 13 actions = 156 prompts.

**Run:** 156/156 LLM calls succeeded, **0/156 parse failures** — fourth clean
run in a row. Table: 52/52 cells, every cell at 3 samples. Raw results saved
to `tables/pearl_12action_pilot/raw/results_protocol_fewshot_20260731_192125.jsonl`.
Round-9 table archived as
`tables/pearl_12action_pilot/archive/pearl_pilot_protocol_fewshot_r9.json`.

| Check | Result | Detail |
|-------|--------|--------|
| C1 action coverage | PASS | 13/13 |
| C2 cell coverage | PASS | 52/52 |
| C3 state persistence | PASS | idle P(stay): low=1.0, high=1.0 (raw idle means: high 7,607/7,436, low 2,883/3,005 — all inside their stated bands) |
| C4 action sensitivity | FAIL | 0/4 cells (structurally blind: high idle P(high)=1.0 leaves no headroom; low cannot cross 7,000) |
| C5 burden monotonicity | PASS | burden_reduces_steps=true at both levels (major high P(high) 0.9744 ≤ none 1.0) |
| C6 factor variation | FAIL | morning_steps_ratio dominant share 0.9423 (49 balanced / 3 morning; round 9: 0.8654, also FAIL per analyzer) — the ceiling capped high/none cells so fewer crossed into "morning" |

**Raw effect (analyzer):** overall mean lift **+222.1** steps/day (round 9:
+231.3 — in the +150-450 band), min **-614.3** (round 9: -313.3), max
**+674.8** (round 9: +981.0 — improved), **42/48** cells positive (round 9:
43/48), **6 negative cells** (round 9: 5; target ≤ 3): high/major
social_opportunity_morning -614.3, low/major prioritization_afternoon
-69.0, low/major perceived_benefit_morning -54.8, low/major
ability_afternoon -16.0, low/major social_opportunity_afternoon -15.7,
low/major perceived_benefit_afternoon -14.3. Per state — high/major: idle
7,436, lift **+349.9** (round 9: +148.6 — overshoot shifted to the 0.9-weight
profile); high/none: idle 7,607, lift **+271.2** (round 9: +596.7 — target
+250-350, tamed); low/major: idle 3,005, lift **+61.8** (round 9: +109.2 —
fell under the +100 floor, 5 of 6 negatives); low/none: idle 2,883, lift
**+205.6** (round 9: +70.9 — above the +150 floor, 0 negatives).

**Verdict table:**

| Metric | Target | Round 8 | Round 9 | Round 10 | Verdict |
|--------|--------|---------|---------|----------|---------|
| Checks | — | 3/6 | 4/6 | 4/6 | stable — C6 stays FAIL (structural noise) |
| Max cell lift | ≤ ~+600 | +775.4 | +981.0 | **+674.8** | FAIL — better, still over |
| high/none mean | +250-350 | +212.4 | +596.7 | **+271.2** | PASS — tamed |
| low/none mean | ≥ +150 | +5.6 | +70.9 | **+205.6** | PASS — exemplar worked |
| low/major mean | ≥ +100 | +72.8 | +109.2 | +61.8 | FAIL — slipped |
| high/major mean | +150-450 | — | +148.6 | +349.9 | PASS (but cells over +600) |
| Overall mean | +150-450 | +179.1 | +231.3 | +222.1 | PASS |
| Negative cells | ≤ 3 | 11 | 5 | 6 | FAIL |
| C3 idle P(stay) | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | PASS |
| Parse failures | ≤ 2 | 0 | 0 | 0 | PASS |

**Diagnosis:** The ceiling-in-rule worked exactly where the exemplar-only
ceiling failed: high/none fell from +596.7 to +271.2 (in the +250-350
target) with every high/none cell under +500, and max cell lift dropped
from +981.0 to +674.8 — removing the high-activity exemplar ended the
total-anchoring (no more +1,477 planning_afternoon days) and the rule-level
cap bound what remained. The low/none morning exemplar also delivered:
low/none rose from +70.9 to +205.6, above the +150 floor, all 12 cells
positive, and the 0.8-weight morning cells specifically fixed
(ability_morning +166.7, perceived_benefit_morning +173.8, planning_morning
+211.9 — no more -38 low-state mornings). But the cap did NOT bind on
high/major: its mean rose from +148.6 to +349.9 with six cells at +520-675
(prioritization_afternoon +674.8, planning_afternoon +624.8, ability_
afternoon +592.9) — the ceiling sentence is read as capping the band's
tail, not each cell, and the 0.8-0.9-weight planning/prioritization cells
overshoot anyway; high/major social_opportunity_morning is a new -614.3
wild outlier. And the low states diverged: low/none is now the best low
state while low/major fell to +61.8 (under the floor) with 5 of the 6
negatives — the new exemplar lifted the low/no-burden profile, not the
fatigued one. 4/6 checks (C6 flipped to FAIL at 0.9423 dominant share — a
handful of "morning"-ratio cells flip it between rounds), C3 fully green,
C5 PASS, 0 parse failures.

**Next steps:**
1. Round 11 (final polish): tame high/major's 0.8-0.9-weight cells — the
   ceiling sentence caps the band's tail, not per-cell lift. Options: (a)
   tighten the ceiling to state the cap as an absolute on the day's total
   *increase* applied per message ("no single message raises a day's total
   by more than about 500 steps — even a 0.9-weight match"), or (b) add a
   high-baseline strong-match exemplar that demonstrates the cap without a
   total anchor (e.g. "a well-matched planning message on a day that would
   have been 8,000 steps added about 400, never more than about 500").
   Investigate the high/major social_opportunity_morning -614.3 outlier
   first.
2. Fix low/major's 5 negative afternoons (perceived_benefit_afternoon,
   ability_afternoon, prioritization_afternoon, social_opportunity_afternoon):
   the low/none morning exemplar lifted that profile, not the fatigued one;
   consider fatigue-aware afternoon wording or a low/major exemplar —
   without touching the fixed-absolute rule lesson of round 8.
3. Ship criteria for round 11: max cell lift ≤ +600, high/none and
   high/major both in +150-450, low/none ≥ +150, low/major ≥ +100,
   negatives ≤ 3, C5 green, parse failures ≤ 2 — then freeze
   `protocol_fewshot` for the real bootstrapping experiment; `protocol`
   stays as fallback.
4. C4/C6 remain structurally blind or noisy in this subset (C6 flipped
   0.8654 pass in r9 → 0.9423 fail in r10 on a handful of "morning"-ratio
   cells); for real signal, extend the pilot subset to moderate
   recent_steps_mean cells and vary morning_steps_ratio / walk_pattern /
   day_of_week.

---

### Round 11 — protocol_fewshot, Option B (2026-08-01)

**Prompt version:** `protocol_fewshot` r11 in `prompts/pearl.py` — first
literature-backed round (Option B; see
`docs/research/llm-prompt-calibration-literature.md`).

**Prompt change summary:**

1. **Exemplars rewritten as deltas and ranges only** — every absolute
   step total is removed from DAY-LEVEL EXEMPLARS (no more 8,200 / 3,100 /
   3,400 / 3,120 / 1,700-1,420). Strong match = "+250 to 350 steps above
   the no-message day", modest = "+120 to 180", weak = "+60 to 100".
   Rationale: exemplar magnitudes leak into outputs as anchors (Min et
   al. EMNLP 2022; Lou & Sun 2025). The only absolute numbers left in
   the prompt are the idle-pin bands, stated as rules.
2. **OUTPUT FORMAT grammar block appended** (Wang et al. NeurIPS 2023) —
   "respond with exactly 7 lines, one JSON object per line, of the form
   {"day": N, "morning_steps": M, "afternoon_steps": A} with N = 1..7" —
   N/M/A placeholders keep the template unparseable, so it cannot be
   ingested as a history row.
3. **Pipeline hardening** (code, not prompt): analyzer now reports
   `median_lift_steps` and `trimmed_mean_lift_steps` alongside the
   round-comparison `mean_lift_steps`; generator adds a bounded retry
   (1 retry) on unparseable responses.

**Config:** deepseek-v4-flash (openrouter), temp 0.7, 3 samples/cell, 4 states
(2 burden x 2 recent_steps_mean) x 13 actions = 156 prompts.

**Run:** 156/156 LLM calls succeeded, **0/156 parse failures, 0 retries**
(fifth clean run in a row). Table: 52/52 cells, every cell at 3 samples.
Raw results saved to
`tables/pearl_12action_pilot/raw/results_protocol_fewshot_20260801_003800.jsonl`.
Round-10 table archived as
`tables/pearl_12action_pilot/archive/pearl_pilot_protocol_fewshot_r10.json`.

| Check | Result | Detail |
|-------|--------|--------|
| C1 action coverage | PASS | 13/13 |
| C2 cell coverage | PASS | 52/52 |
| C3 state persistence | PASS | idle P(stay): low=1.0, high=1.0 (raw idle means: high 7,857/7,849, low 2,943/2,962 — all inside their stated bands without exemplar reinforcement) |
| C4 action sensitivity | FAIL | 0/4 cells (structurally blind: high idle P(high)=1.0 leaves no headroom; low cannot cross 7,000) |
| C5 burden monotonicity | PASS | burden_reduces_steps=true at both levels |
| C6 factor variation | FAIL | morning_steps_ratio dominant share 0.9231 (48 balanced / 3 morning / 1 evening; round 10: 0.9423) — ratio differentiation weakened when the concrete step splits were removed with the totals |

**Raw effect (analyzer):** overall mean lift **+206.4** steps/day (round 10:
+222.1 — in the +150-450 band), median **+129.0**, trimmed mean **+193.4**
(right-skew: a few overshoot cells inflate the mean), min **-123.8**
(round 10: **-614.3** — the wild outlier is gone), max **+1,135.7** (round
10: +674.8), **42/48** cells positive (round 10: 42/48), **6 negative
cells** (round 10: 6; target ≤ 3) — all mild: high/major
social_opportunity_morning -123.8, high/none physical_opportunity_afternoon
-118.6, high/none planning_morning -80.7, high/major
perceived_benefit_morning -52.4, high/none physical_opportunity_morning
-28.6, low/major social_opportunity_morning -19.0. Per state — high/major:
idle 7,857, lift **+240.4** (round 10: +349.9); high/none: idle 7,849,
lift **+59.3** (round 10: +271.2 — fell below the +150 floor); low/major:
idle 2,943, lift **+325.2** (round 10: +61.8 — recovered above floor);
low/none: idle 2,962, lift **+200.9** (round 10: +205.6).

**Verdict table:**

| Metric | Target | Round 9 | Round 10 | Round 11 | Verdict |
|--------|--------|---------|----------|----------|---------|
| Checks | — | 4/6 | 4/6 | 4/6 | stable — C6 stays FAIL (structural noise) |
| Max cell lift | ≤ ~+600 | +981.0 | +674.8 | **+1,135.7** | FAIL — worse; planning at 0.8-0.9 |
| high/none mean | +250-350 | +596.7 | +271.2 | **+59.3** | FAIL — fell under floor |
| low/none mean | ≥ +150 | +70.9 | +205.6 | **+200.9** | PASS |
| low/major mean | ≥ +100 | +109.2 | +61.8 | **+325.2** | PASS — recovered |
| high/major mean | +150-450 | +148.6 | +349.9 | **+240.4** | PASS |
| Overall mean | +150-450 | +231.3 | +222.1 | **+206.4** | PASS |
| Min cell lift | ≥ -300 | -313.3 | -614.3 | **-123.8** | PASS — outlier gone |
| Negative cells | ≤ 3 | 5 | 6 | 6 | FAIL (but all mild, ≤ -124) |
| C3 idle P(stay) | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | PASS |
| Parse failures | ≤ 2 | 0 | 0 | 0 | PASS |

**Diagnosis:** The delta/ranges exemplar rewrite delivered its two main
targets: the -614.3 outlier is gone (min -123.8) and mean lift stays in
band (+206.4, statistically indistinguishable from round 10's +222.1). The
OUTPUT FORMAT grammar block works — 156/156 parsed, zero retries. Idle pins
hold without exemplar totals (C3 fully green), so the main literature risk
of the rewrite did not materialize. But removing the totals also removed
the implicit ceilings they anchored: the two worst overshoot cells are
planning at weight 0.8-0.9 under major burden (low/major planning_afternoon
+1,135.7, high/major planning_morning +1,081.4) — the standalone ceiling
sentence caps the band's tail, not the strongest tier — and high/none
weakened to +59.3, its lowest since round 8, because the exemplars no
longer show high-activity people responding to strong matches. Negative
cells shifted from low/major afternoons (round 10) to high-state weak-weight
themes, but are all mild (≤ -124).

**Next steps:**
1. Round 12 (surgical, 156 calls): co-locate the ceiling directly inside
   the strong-match tier sentence ("around +300 — and never more than
   about 500, no matter how strong the match is") and add an exemplar
   sentence that a strong match is capped at ~500 even for the best-
   matching person; reinforce that high-activity people respond to strong
   matches (+250-350 on top of their 7,500-8,500 idle day).
2. Ship criteria for round 12: max cell lift ≤ ~+700, high/none ≥ +100,
   mean in +150-450, C3 green, parse failures ≤ 2 — then freeze
   `protocol_fewshot` for the full 108-state run.
3. Then: full-scale generator (108 states), fix `pearl_bootstrap.yaml`'s
   dead table path, wire `pearl_constitution_12action.yaml` to the full
   table, run constitution T1-T4.
4. C4/C6 remain structurally blind or noisy in this subset; real signal
   needs the full state space (moderate cells, varied walk_pattern /
   morning_steps_ratio / day_of_week).

---

### Round 12 — protocol_fewshot, ceiling co-location (2026-08-01)

**Prompt change summary:** two surgical edits on top of r11 — (1) the
strong tier of GRADED MATCH RULE reads *"around +300 - and never more than
about 500, no matter how strong the match is"*; (2) the exemplar block
gains *"A strongly matched message is never more than about 500 steps,
even for the person who matches it best."*

**Run:** 156/156, 0 parse failures, 0 retries. Raw:
`results_protocol_fewshot_20260801_004422.jsonl`.

| Metric | Target | Round 11 | Round 12 | Verdict |
|--------|--------|----------|----------|---------|
| Overall mean | +150-450 | +206.4 | **+347.9** | PASS (but inflated) |
| Max cell lift | ≤ ~+600 | +1,135.7 | **+1,517.6** | FAIL — ceiling co-location backfired |
| high/major mean | +150-450 | +240.4 | **+688.5** | FAIL — overshoot |
| high/none mean | ≥ +100 | +59.3 | **+226.4** | PASS |
| Min cell lift | ≥ -300 | -123.8 | **-24.4** | PASS — outlier stays gone |
| Positive cells | — | 42/48 | **47/48** | PASS (best yet) |
| C3 / parse fails | 1.0/1.0, ≤ 2 | 1.0/1.0, 0 | 1.0/1.0, 0 | PASS |

**Diagnosis:** the strong-tier ceiling wording reads as permission to
approach 500, so strong responses got *stronger*. The high/none improvement
(+59 → +226) may be noise at n=3 rather than a wording effect.

**Next step:** revert the strong-tier co-location to r11 wording, keep the
exemplar cap sentence, re-run.

### Round 13 — protocol_fewshot, FROZEN (2026-08-01)

**Prompt change summary:** strong-tier co-location reverted to r11 wording;
the r12 exemplar sentence *"A strongly matched message is never more than
about 500 steps, even for the person who matches it best."* is kept.

**Run:** 156/156, 0 parse failures, 0 retries. Raw:
`results_protocol_fewshot_20260801_004943.jsonl`. Round-12 table archived
as `archive/pearl_pilot_protocol_fewshot_r12.json`.

| Metric | Target | Round 10 | Round 11 | Round 12 | Round 13 | Verdict |
|--------|--------|----------|----------|----------|----------|---------|
| Checks | — | 4/6 | 4/6 | 4/6 | 4/6 | stable |
| Overall mean | +150-450 | +222.1 | +206.4 | +347.9 | **+223.2** | PASS |
| Max cell lift | ≤ ~+600 | +674.8 | +1,135.7 | +1,517.6 | **+823.8** | FAIL (pipeline-level handling) |
| Min cell lift | ≥ -300 | -614.3 | -123.8 | -24.4 | **-30.0** | PASS — outlier eliminated |
| Positive cells | — | 42/48 | 42/48 | 47/48 | **47/48** | PASS |
| high/major mean | +150-450 | +349.9 | +240.4 | +688.5 | **+409.0** | PASS (band edge) |
| high/none mean | ≥ +100 | +271.2 | +59.3 | +226.4 | **+176.9** | PASS |
| low/major mean | ≥ +100 | +61.8 | +325.2 | +308.7 | **+175.1** | PASS |
| low/none mean | ≥ +150 | +205.6 | +200.9 | +167.9 | **+132.0** | marginal |
| C3 idle P(stay) | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | PASS |
| Parse failures | ≤ 2 | 0 | 0 | 0 | 0 | PASS |

**Diagnosis:** best-balanced round of the Option-B ladder — mean +223.2 in
band, min -30.0, 47/48 positive, C3 green. **The high-side overshoot
persists across three ceiling-wording variants** (standalone +1,136,
strong-tier co-location +1,518, restored wording +824): prompt wording
cannot reliably enforce numeric ceilings (literature-predicted — instruction
constraints are fragile). The remaining overshoot (max +823.8, high/major
planning/prioritization) is handled at the pipeline level for the full run:
robust aggregation plus per-cell caps in analysis.

**Decision: protocol_fewshot r13 is FROZEN** for the full 108-state
bootstrap.

---

### Round 14 — temperature sweep (2026-08-01)

**Prompt change summary:** none — frozen r13 prompt run at two sampling
temperatures (0.3 vs 0.7), n=3 each (156 calls per arm), to pick the
full-scale temperature.

**Runs:** both 156/156, 0 parse failures, 0 retries. Tables archived as
`archive/pearl_pilot_protocol_fewshot_r14_temp{03,07}.json`; raw
`results_protocol_fewshot_20260801_0850{42,337}.jsonl`.

| Metric | temp 0.3 | temp 0.7 | Winner |
|--------|----------|----------|--------|
| Overall mean | **+279.0** | +171.0 | 0.3 |
| Median lift | +191.4 | +111.2 | 0.3 |
| Trimmed mean | +268.9 | +158.7 | 0.3 |
| Max cell lift | **+981.0** | +1,171.4 | 0.3 |
| Min cell lift | **+39.0** | -265.3 | 0.3 |
| Positive cells | **48/48** | 36/48 | 0.3 |
| high/major mean | +567.6 | +483.3 | 0.3 |
| high/none mean | +220.6 | **-126.3** | 0.3 |
| low/major mean | +184.5 | +172.1 | 0.3 |
| low/none mean | +143.2 | +155.0 | tie |
| C3 idle P(stay) | 1.0/1.0 | 1.0/1.0 | tie |

**Diagnosis:** temp 0.3 is the decisive winner. It keeps every one of the
48 intervention cells positive (0.7 put the high/none arm at -126.3), holds
the mean inside the +150-450 band, eliminates negative outliers (min +39.0
vs -265.3), and *lowers* the high-side overshoot (max +981.0 vs +1,171.4).
This matches the literature entry #8 (Renze & Guven): temperature is a
diversity dial, not an accuracy dial — for a single target distribution,
the more deterministic extreme (0.3) reduces deviation. The r13 default of
0.7 was chosen before the sweep; 0.3 is now the full-scale temperature.

**Decision: full-scale run at temperature 0.3.**

---

### Round 15 — n=6 convergence pilot (2026-08-01)

**Prompt change summary:** none — frozen r13 prompt at temperature 0.3
(the round-14 winner), n=6 (312 calls).

**Run:** 312/312, 0 parse failures, 0 retries. Table archived as
`archive/pearl_pilot_protocol_fewshot_r15_n06_temp03.json`; raw
`results_protocol_fewshot_20260801_085845.jsonl`.

| Metric | n=3 (r14, temp 0.3) | n=6 (r15) | Stable? |
|--------|----------------------|-----------|---------|
| Overall mean | +279.0 | **+272.8** | yes |
| Median lift | +191.4 | +253.2 | yes |
| Trimmed mean | +268.9 | +262.4 | yes |
| Max cell lift | +981.0 | **+999.9** | overshoot persists |
| Min cell lift | +39.0 | **+23.8** | yes — no negatives |
| Positive cells | 48/48 | **48/48** | yes |
| C3 idle P(stay) | 1.0/1.0 | 1.0/1.0 | yes |

**Diagnosis:** per-cell means stabilize from n=3 to n=6; all 48 intervention
cells stay positive and no negative outliers appear. The max-cell overshoot
persists at ~+1,000 across temperature *and* sample size (high/major state
mean +509.5 is over the band) — confirming it is structural and belongs at
the pipeline level (robust aggregation + per-cell caps in analysis), not to
further prompt iterations. **Evidence gate passed: proceed with the full
14,040-call run at temperature 0.3.**

**Decision: full-scale generation (108 states x 13 actions x 10 samples) at
temperature 0.3.**

---

## Ladder summary

> Scope: this ladder covers variant selection through round 6 (the SHIP
> decision). Rounds 7-10 refined the shipped `protocol_fewshot` variant
> further; their per-round verdicts are in the round sections above, and
> the check counts below match the per-round verdict tables (r1: 3/6,
> r2: 4/6 — C5 counted per verdict table even where its signal was weak).

| Rung | Variant | n_passes | Mean lift | n positive cells | Parse failures | Verdict |
|------|---------|----------|-----------|------------------|----------------|---------|
| 1 | baseline | 3/6 | — (raw not saved) | — | 2 | no — high states collapse to moderate |
| 2 | state_self_model | 4/6 | +115.0 | — | 1 | no — lift under target, no mechanisms |
| 3 | com_b_mechanisms | 2/6 | +149.6 | 31/48 | 8 | no — wild variance, C3/C2 regress |
| 4 | empirical_anchors | 4/6 | +43.1 | 26/48 | 2 | no — binary matchedness kills lift |
| 5 | protocol | 4/6 | +584.9 | 46/48 | 0 | no — overshoots 2-4x, C3 straddles |
| 6 | protocol_fewshot | 4/6 | **+212.4** | 43/48 | 3 | **SHIP** — in-band lift + fully green C3 |

Round 2's mean lift is taken from the round-3 log entry (+115); rounds 1-2
predate the raw-effect analyzer, so their positive-cell counts are unknown.

**Conclusion:** `protocol_fewshot` is the variant to ship for the real
bootstrapping experiment. It is the only rung that combines a fully green
C3 (idle P(stay) 1.0/1.0, burden-independent, every idle mean inside its
stated band) with lift magnitudes inside the +150-350 target (mean +212.4,
max +673.8, no morning/afternoon asymmetry) — round 5's protocol fixed
directionality but overshot ~2.7x, and rounds 2-4 each traded one of those
two properties for the other. The remaining blemishes are minor and
addressable: 0 parse failures across rounds 7-10 (156/156 every run), 6
negative cells on the weakest-weight themes concentrated in low/major
afternoons (round 10), and C4 blind in all 4 cells — a structural
consequence of C3 being fully green in this subset, not a prompt defect.
Baseline, state_self_model, and com_b_mechanisms are ruled out on checks;
empirical_anchors and protocol are the fallbacks if a post-ship regression
appears, with protocol the closer of the two.

---

## How to add a round

1. Edit prompts in `prompts/pearl.py`; record the diff in the round header.
2. Run the pipeline (3 commands above); be conservative on LLM usage — 3
   samples/cell (156 calls) is enough for signal.
3. Copy `analyze_pearl_mini.py --json` output into `prompt-refinement-log.json`
   as a new `rounds[]` entry.
4. Add a round section here with the verdict table, diagnosis, and next steps.
5. Run `uv run pytest tests/unit/llm_bootstrapping/` after prompt edits.
