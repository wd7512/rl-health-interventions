# Subphase 1C: User Simulation Engine (Rule-Based)

**Status:** `[ ]` Not started

**Dependencies:** 1A, 1B, Dataset Exploration

**Gate requires:** Dataset exploration report completed AND rule-based profiles working

**Parallelises with:** 1D (after 1A/1B gates clear)

---

## Gate Checklist

- [ ] Configurable user profiles instantiated from config (goal-driven, social responder, resistant, stable maintainer)
- [ ] Behavioural response model produces plausible step-count changes given state + action
- [ ] Backlash/fatigue mechanism implemented (burden threshold, response decay)
- [ ] Dataset exploration report evaluating All of Us and UK Biobank for simulator training exists in docs/
- [ ] `uv run pytest` passes for all 1C tests
- [ ] `uv run ruff check` and `uv run ty check` pass

---

## TDD Checklist

- [ ] Write test for each behavioural archetype producing expected response direction (goal-driven increases steps after goal reminder, resistant shows flat response) *before* implementing models
- [ ] Write test for fatigue accumulation and decay: burden increments on intervention, resets on no-action, response probability decays past threshold *before* implementing burden logic

---

## Key Interfaces

### `UserProfile`
```python
class UserProfile(BaseModel):
    archetype: Literal["goal_driven", "social_responder", "resistant", "stable_maintainer"]
    parameters: dict[str, float]
```

### `ResponseModel`
```python
class ResponseModel(ABC):
    @abstractmethod
    def response(self, state: StateView, action: int, profile: UserProfile) -> float
```

`ResponseModel` returns a scalar response magnitude (e.g. Δsteps), unlike
`TransitionModel` which returns a full `StateView`. The transition model
applies the response to produce the next state; the response model only
predicts the behavioural outcome.

---

## Dataset Exploration

✅ **COMPLETE** — see [`sources/data_sources.md`](sources/data_sources.md) for the full feasibility study.

### Additional: JITAI Trial Datasets

See [`sources/additional_data_sources.md`](sources/additional_data_sources.md) for a survey of HeartSteps V1/V2 and other accessible benchmarks. Key findings for 1C:

- **HeartSteps V1** (42 days, ~50 participants) — the best available source of *action → response* mappings. Real engagement decay curves for burden calibration.
- **HeartSteps V2** (90 days, ~97 participants) — RL-in-the-loop data with Thompson Sampling. Provides non-stationary engagement patterns over 3 months.
- **WISDM / ExtraSensory** — publicly available today. Use to validate the data pipeline on real sensor data.

**Recommendation:** Start HeartSteps data access requests early (weeks to months). Use WISDM/ExtraSensory for pipeline testing in the meantime.

| Dataset | Source | Size | Access |
|---------|--------|------|--------|
| All of Us Fitbit | Nature Medicine 2026 | 59K participants, 14 years | Researcher workbench (cloud-only) |
| UK Biobank Accelerometer | npj Digital Medicine 2024 | 100K participants, 7 days each | Application required (4-8 weeks) |

**Conclusion:** Both datasets require institutional applications. Phase 1 uses synthetic data generators parameterised from published statistics. Real data integration is Phase 2.

**Key findings:**
- Between both datasets, most wearable MDP variables are covered (steps, HR, sleep, sedentary, time_of_day)
- Gaps are state variables (goal_progress, burden) — these are simulated, not from data
- Body measures need EHR linkage (All of Us has 46% linked)
- UK Biobank has two data levels: daily summaries and raw 100Hz accelerometer — treat separately

---

## Blocking Risks

- **Real data timeline:** Both datasets require 4-8 week institutional applications. Phase 1 is designed around synthetic data — this is now the confirmed approach, not a fallback.
- **Archetype validity:** The 4 archetypes are theoretical. No guarantee they match real user behaviour. Document this as an explicit assumption.

---

## Logging & Error Handling

See canonical setup in [`code_design.md`](code_design.md#logging--error-handling-canonical).

Subphase-specific concerns for 1C (user simulation):

- **DEBUG events (per step):** user profile id, archetype, current motivation,
  current burden, response magnitude. Same caveat as 1B — disabled by default.
- **Response model failures:** If `ResponseModel.response()` returns NaN,
  Inf, or a value outside the expected range (e.g., step delta > 10K),
  log WARNING with the offending value and clamp to the expected range.
  NaN/Inf in the simulator breaks the entire downstream pipeline.
- **State evolution errors:** If a user state's motivation or
  engagement drops below 0 or above 1, log WARNING and clamp. These are
  bounded [0, 1] quantities by construction. Burden is a non-negative
  integer (`ℤ≥₀`) — never clamped to [0, 1].
- **Archetype sampling:** INFO at episode start: "User synthetic_042 drawn
  from archetype 'goal_driven' (seed 42)". Useful for offline
  analysis of per-archetype performance.
- **Non-stationarity check:** Every 50 episodes, log a DEBUG line with the
  distribution of motivation and burden values across the active user pool.
  Catches a buggy simulator that's stuck at one state.

Related 1C tests:
- `tests/unit/simulation/test_response_clamp.py` — out-of-range response
  is clamped, not raised.
- `tests/unit/simulation/test_state_bounds.py` — motivation/engagement never
  escape [0, 1]; burden is non-negative integer.
- `tests/integration/simulation/test_archetype_sampling.py` — 1000 user
  draws produce the expected archetype distribution.
