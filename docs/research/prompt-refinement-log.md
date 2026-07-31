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

**Summary:** 2/6 checks pass. Pipeline works end-to-end; prompt behaviour does
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

**Summary:** 3/6 checks pass (round 1: 2/6). C3 fully fixed; C4/C6 fail for
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

**Summary:** 2/6 checks pass (round 2: 3/6). Mean intervention lift moved
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

## Ladder summary

| Rung | Variant | n_passes | Mean lift | n positive cells | Parse failures | Verdict |
|------|---------|----------|-----------|------------------|----------------|---------|
| 1 | baseline | 2/6 | — (raw not saved) | — | 2 | no — high states collapse to moderate |
| 2 | state_self_model | 3/6 | +115.0 | — | 1 | no — lift under target, no mechanisms |
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
addressable: 3 prose-preamble parse failures (no cell lost), 5 negative
cells on the weakest-weight themes (social_opportunity afternoons), and C4
blind in all 4 cells — a structural consequence of C3 being fully green in
this subset, not a prompt defect. Baseline, state_self_model, and
com_b_mechanisms are ruled out on checks; empirical_anchors and protocol
are the fallbacks if a post-ship regression appears, with protocol the
closer of the two.

---

## How to add a round

1. Edit prompts in `prompts/pearl.py`; record the diff in the round header.
2. Run the pipeline (3 commands above); be conservative on LLM usage — 3
   samples/cell (156 calls) is enough for signal.
3. Copy `analyze_pearl_mini.py --json` output into `prompt-refinement-log.json`
   as a new `rounds[]` entry.
4. Add a round section here with the verdict table, diagnosis, and next steps.
5. Run `uv run pytest tests/unit/llm_bootstrapping/` after prompt edits.
