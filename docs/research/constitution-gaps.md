# PEARL Constitution — Gap Map (Layer-2 fixes)

**Status:** Pre-execution, grounded in the check scripts, reference data, and
r13 prompt-ladder results. Fixes below are applied in Phase 3 of
`plans/option-b-full-scale-pearl-bootstrap.v2.md` and validated on the full
108-state × 13-action bootstrap table.

---

## Gap 1 — T2.3 ordering is stricter than the evidence supports

**Where:** `scripts/pearl_constitution/run_distribution_check.py:86-117`
(`check_t2_3_effect_size_ordering`).

**Current check:** `RL ≥ Fixed ≥ Random > Control` (strict chain).

**Evidence:**
- `docs/research/reference/pearl_reference.json` effect sizes: RL vs Fixed
  Δ=238, RL vs Random Δ=218, RL vs Control Δ=296 (1 month). Fixed and Random
  both sit well below RL and near each other.
- r13 pilot ladder: RL dominates; Fixed/Random/Control cluster far below with
  no reliable ordering between them.

**Intended check (gap map):** `RL > (Fixed ≈ Random ≈ Control)` — RL must
strictly beat all three, but no ordering is asserted between Fixed, Random,
and Control.

**Fix:** test `RL mean > max(Fixed, Random, Control)`; drop the
`Fixed ≥ Random > Control` chain.

---

## Gap 2 — T2.2 band ambiguous (150-450 vs 210-296)

**Where:** `scripts/pearl_constitution/run_distribution_check.py:58-83`
(`check_t2_2_effect_size_magnitude`).

**Current check:** `150 ≤ Δ ≤ 450` (a wide sanity band).

**Evidence for the narrower band:**
- `docs/research/reference/pearl_reference.json` `effect_sizes`:
  - `rl_vs_control_1mo`: Δ=296
  - `rl_vs_control_2mo`: Δ=210
  - `rl_vs_random_1mo`: Δ=218
  - `rl_vs_fixed_1mo`: Δ=238
  - `gee_sustained`: Δ=208
  - Observed range across all reference deltas: **210-296**; across
    1-month deltas only (the T2.2 horizon): **218-296**.
- r13 pilot mean lift (+223.2) sits inside 210-296.

**Decision:** the 150-450 band is deliberately wide as a structural guard
against degenerate tables. The empirical band encodes the reference. The
check derives its band at runtime from `load_reference()` 1-month deltas
(`rl_vs_control_1mo` Δ=296, `rl_vs_random_1mo` Δ=218, `rl_vs_fixed_1mo`
Δ=238 → **218-296**); 2-month and sustained deltas (210, 208) are excluded
because T2.2 tests the 1-month horizon. 150-450 remains the documented
fallback floor when the reference carries no 1-month effect sizes.
**Verify before changing:** if the full-scale table produces a healthy mean
lift in 218-296, keep the derived band; if it lands outside but inside
150-450 (e.g. the r13 ceiling behavior), keep 150-450 and log why.

**Fix:** derive the band from `load_reference()` 1-month deltas where
possible; keep 150-450 as the documented fallback floor.

---

## Gap 3 — μ baseline 5,618 vs 5,580 discrepancy

**Where:**
- `docs/research/reference/pearl_reference.json` → `mean_baseline_steps:
  5618.2` (the source of truth; T1.1/T2.1 read it via `load_reference()`).
- `scripts/pearl_constitution/utils.py:38-42`
  (`_RECENT_STEPS_MEAN_MIDPOINT`) — comment cites "PEARL Table 3 (baseline
  mean=5,580, SD=1,499)".

**Resolution:** 5,618.2 is the persisted reference statistic and is what the
checks actually compare against. The 5,580 figure in the comment is the
midpoint the state-factor bins were calibrated to and is a rounding/derivation
artifact. **Fix:** correct the comment in `utils.py` to cite 5,618.2 and note
that check constants derive from `load_reference()`, not the hardcoded comment.

---

## Gap 4 — persona_name wiring skips 12-action table override

**Where:** `scripts/pearl_constitution/utils.py:278-326`
(`load_constitution_config`).

**Current behavior:** for the 12-action config (`steps_per_day == 1`), the
persona table override is skipped with a log line — personas are handled by
COM-B agent scores instead of transition tables.

**Intended behavior (gap map):** this is correct for the 12-action config
(`tables/pearl_12action/`), because PEARL personas are a COM-B construct, not
a transition-table variant. The Sprint-1 persona tables
(`tables/persona/*_deepseek-v4-flash`) remain reachable only through the
original 4-action config path.

**Fix:** document this as intended in the docstring (done) and add a guard test
asserting the 12-action config never inherits a persona table_dir.

---

## Gap 5 — T4.2 persona-collapse is vacuous

**Where:** `scripts/pearl_constitution/run_stress_tests.py:77-90`
(`check_t4_2_persona_collapse`).

**Current behavior:** always returns "Skipped: requires multi-persona data with
identity transitions" and marks PASS.

**Intended:** a real check — multi-persona identity-transition ANOVA — requires
identity-transition tables per persona that do not exist in this repo (Sprint-1
personas use real LLM tables; PEARL 12-action uses COM-B scores). The skip is
therefore a **documented limitation**, not a silent pass.

**Fix:** keep the skip but emit it at WARNING (not as a green PASS), and state
the limitation explicitly in the output. Optionally gate on an env flag
`PEARL_T4_2_AVAILABLE=1` to run the real check when identity tables exist.
