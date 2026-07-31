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

## How to add a round

1. Edit prompts in `prompts/pearl.py`; record the diff in the round header.
2. Run the pipeline (3 commands above); be conservative on LLM usage — 3
   samples/cell (156 calls) is enough for signal.
3. Copy `analyze_pearl_mini.py --json` output into `prompt-refinement-log.json`
   as a new `rounds[]` entry.
4. Add a round section here with the verdict table, diagnosis, and next steps.
5. Run `uv run pytest tests/unit/llm_bootstrapping/` after prompt edits.
