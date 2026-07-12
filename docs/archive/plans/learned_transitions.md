# Learned Transitions — Implementation Plan

**Date:** 2026-06-17
**Mode:** Implementation plan (pre-coding gate for adding data-driven transition models)
**Goal:** Replace the hand-specified rule-based transition matrix with a learned residual ridge regression `Δs = s_{t+1} - s_t ≈ Ridge(s, a)`, falsified against a no-op baseline. NN upgrade is a gated follow-up, not a default.

---

## Background

The MVP simulator (`config/rule_based.yaml`, `src/rl_health_interventions/transitions/rule_based.py`) uses a static, hand-typed probability table. The canonical design doc (`docs/decisions/initial_design.tex:140-178`) describes richer dynamics (continuous state from wearables, action-conditioned response, burden accumulation) that the rule-based path cannot represent without bespoke engineering per dataset.

This plan defines how learned, data-driven transitions enter the framework: which datasets feed them, what model class is used first, what falsifies it, and how the ABC surfaces widen to admit it.

---

## Locked decisions

Captured from a grilling session on 2026-06-17. Each decision is paired with the trade-off that was accepted.

| # | Decision | Accepted trade-off |
|---|---|---|
| Q1 | Primary loaders: **4TU #1** "Collaboratively Setting Daily Step Goals" (verified schema, 235 users × 1-5 sessions, 5-level action, 6-feature state incl. prior-day steps). Backups: 4TU #2 (pre-packaged (s,a,s',r)), 4TU #5 (3.36 GB RL-samples), LifeSnaps. | Throws out the TILES candidate (see "TILES — dropped" below). Smaller and more psychological than physiological; doesn't match the design doc's HR/steps/sleep-heavy state. |
| Q2 | Residual ridge regression `Δs = s_{t+1} - s_t ≈ Ridge(s, a)` as Stage 1. NN as gated Stage 2 upgrade. | Rejects the "binary-then-NN" simplification — no binary stage needed because rich data is available. Removes the cheap data-pipeline validation step. |
| Q3 | Falsification: ridge must beat no-op baseline (`Δŝ = 0`) by ≥20% held-out RMSE **on both** LOUO and within-user temporal split, **on at least one primary feature**. | 20% threshold is arbitrary; selected for simplicity on ~700 transitions (4TU #1). |
| Q4 | Widen existing `TransitionModel` ABC to `(StateView, str) -> StateView`. Add `features: dict[str, float] \| None` to `StateView`. Add `LearnedTransition` as new subclass. `RuleBasedTransition` gets a 3-line signature conformance. | Touches the working MVP — regression fixture must be re-verified after refactor. Avoids type-discriminated dispatch in `Environment`. |
| Q5 | 20% threshold only (no bootstrap CI, no paired tests for v1). | Simpler to compute; less rigorous than statistical significance tests. Can upgrade later. |
| Q6 | Pure numpy closed-form ridge via `np.linalg.solve`. No new runtime dependencies. λ via manual grid search on LOUO. | More code than `sklearn.linear_model.RidgeCV`; honours `AGENTS.md` minimal-deps rule. |
| Q7 | Coefficient sanity check is autocorrelation only: `prior_steps → next_steps` weight must be > 0, `prior_hr → next_hr` weight must be > 0 (where applicable). | Does not test intervention effect direction (would assume an unsupported ground-truth effect). |
| Q8 | Loader strategy decided at impl time: modify existing `load_4tu_step_goals` (which reads the 117-row summary CSV from dataset `6f8e6750`) or add `load_4tu_collaborative_goals` (which reads the 3.6 MB zip from dataset `53f2d238`). Whichever preserves more information. | Two-loader maintenance cost if both kept. |
| Q9 | To resolve ambiguity of autocorrelation check under residual prediction (`w_prior_steps > -1` is weak), refit a separate direct `s_{t+1} = Ridge(s, a)` model and assert its `prior_steps` coefficient > 0. | One extra ridge fit (~10 lines numpy). Strict, unambiguous data-correctness gate. |

---

## Datasets

### Primary: 4TU #1 "Collaboratively Setting Daily Step Goals"

- **Source:** [4TU.ResearchData](https://data.4tu.nl/datasets/53f2d238-77fc-4045-89a9-fb7fa2871f1d/1)
- **Citation:** Dierikx, Albers, Scheltinga, Brinkman (2024). DOI `10.4121/53f2d238-77fc-4045-89a9-fb7fa2871f1d.v1`.
- **License:** CC BY 4.0
- **Size:** 3.6 MB zip (one combined archive with `.csv`, `.ipynb`, `.py`, `.Rmd` analysis code).
- **N:** 235 participants × 1-5 conversational sessions ≈ ~700 transitions.
- **Time coverage:** June – July 2023.
- **State (6 features):** mood, sleep quality, available time, motivation, self-efficacy, prior-day step count.
- **Action (5 levels):** RL-randomised step-goal proposal: −400, −200, 0, +200, +400 (offset from the coach's recommended goal).
- **Response:** participant's chosen goal + next-day self-reported steps.
- **Notes:** Likely a superset of the existing 117-row `load_4tu_step_goals` CSV (which reads from dataset `6f8e6750`, the predecessor "RL to Personalize Daily Step Goals" thesis dataset). The actual zip structure (column names, how sessions link to users) must be inspected at impl time by downloading and unzipping.
- **Limitation flagged:** State is psychological + one physiological feature (prior-day steps). Does not match the design doc's HR/sleep-hours/sedentary-minutes physiological state (`initial_design.tex:412-415`). Update design doc decision log when this is wired in.

### Backups (in order of fallback)

1. **4TU #2 "Dyadic PA Planning with a Virtual Coach"** — Stefan 2023, DOI `10.4121/2796f502-...-v1`, 114 participants × ≤5 sessions, 5.1 MB zip. Ships pre-extracted (state, action, next-state, reward) tuples. State is also psychological (confidence, perceived usefulness, attitude). Cleanest v1 dataset if 4TU #1 proves unworkable.
2. **4TU #5 "RL for Smoking Cessation + PA"** — Albers 2024 v3, DOI `10.4121/21533055.v3`, multi-session daily smokers, 3.36 GB zip. Pre-extracted "RL-samples (states, actions, rewards)". Half the data is smoking-cessation confound. Has RL-in-the-loop action selection in the back half of the study.
3. **LifeSnaps** — Hermans et al. 2023 (Scientific Data), TU Eindhoven, 71 participants × 4 weeks × 4 phases (baseline → low → high → recovery). Phased design is quasi-intervention. Has Fitbit (HR, steps, sleep, calories) + smartphone + EMA. Open access via 4TU.

### TILES — DROPPED

**Status:** Removed from this plan. The dataset is not what the repo's source documents claim.

**Verified facts (from Mundnich et al. 2020, *Scientific Data* 7:354, DOI `10.1038/s41597-020-00655-3`):**

- Actual name: **TILES-2018** ("Tracking IndividuaL performancE with Sensors, year 2018").
- Authors: Mundnich, Booth, L'Hommedieu, Feng, Girault, Lerman, Narayanan et al. (USC / USC ISI; Tiago H. Falk at INRS).
- Lead institution: **USC Keck Hospital** (not Northwestern). "Laura Stroud (Northwestern)" attribution in `docs/sources/additional_data_sources.md:263` is incorrect.
- Hosting: `https://tiles-data.isi.edu/` (not OSF). The OSF link `osf.io/jm6du/` cited in the docs does not resolve to a TILES dataset — the OSF API returns 404 on `/v2/nodes/jm6du/` and `/v2/registrations/jm6du/`, and the page returns unrelated search results.
- Participants: n = 212 (not ~200).
- Duration: 10 weeks (not 12 months).
- Sensors: Fitbit Charge 2 (steps, sleep, HR), OMsignal biometrics garment, Jelly phone audio-features recorder, Owl-in-One Bluetooth hubs, Minew environmental sensors (door motion, humidity, temperature, light).
- **Intervention delivery:** **none.** The paper describes it as observational ("in their natural day-to-day job settings"; "in situ experimental study in a real world workplace"). No mention of interventions, treatments, randomized arms, or delivery logs. EMAs (Ecological Momentary Assessments) are surveys, not agent actions.

**Conclusion:** TILES-2018 has no `a_t` intervention variable and therefore cannot be used to learn `P(s_{t+1} | s_t, a_t)`. The "intervention delivery logs" claim appears in 5 places in `additional_data_sources.md`, 12 places in `simulator-validation.md`, and 2 places in `SUPERVISOR_SUMMARY.md`. The most-directly-wrong section (`additional_data_sources.md:260-284`) is corrected in this PR; downstream references are tagged for follow-up.

---

## Stages and execution order

### Stage 0 — Resolve blockers (4TU only)

Download the 3.6 MB zip from 4TU #1, unzip into `data/4tu_collaborative/`, and inspect:

- File tree (which CSVs are present, excluding the `.ipynb`/`.py`/`.Rmd` analysis scripts).
- The per-session/per-user transition CSV: column names, row granularity, how sessions link back to `user_id`.
- Confirm the row count is in the hundreds (per the paper's reported 235 participants × 1–5 sessions).

### Stage 1 — Widen StateView + ABCs (backward-compatible refactor)

**Files to touch:**

| File | Change |
|---|---|
| `src/rl_health_interventions/state.py` | Add `features: dict[str, float] \| None = None` (frozen dataclass; `field(default=None)`). |
| `src/rl_health_interventions/transitions/_base.py` | Widen ABC signature to `transition(state: StateView, action: str) -> StateView`. |
| `src/rl_health_interventions/transitions/rule_based.py` | 3-line conformance: extract `state.activity`; return `dataclasses.replace(state, activity=next)`. Body of `transition()` otherwise unchanged. |
| `src/rl_health_interventions/rewards/_base.py` | Widen ABC signature to `reward(state: StateView, action: str, step_idx: int) -> tuple[float, bool]`. |
| `src/rl_health_interventions/rewards/compound.py` | Extract `state.activity` for current lookup path. |
| `src/rl_health_interventions/environment.py` | Pass `self._current_state` to transition and reward; use `dataclasses.replace()` to merge `day` / `step_of_day` from transition-returned StateView. |
| `src/rl_health_interventions/config/schemas.py` | Add schema validation for `features` field (dict[str, float] \| None). Relax the cross-reference validator to admit non-rule-based transition types. Remove the `CompoundReward` `NotImplementedError` short-circuit at schema-ref mode if it blocks the learned path. |

**Verification:**

- The regression fixture `tests/fixtures/mvp_expected_rewards.json` (currently pinned at `{"thompson_sampling": 165.0}` for `config/rule_based.yaml` at seed 42) must produce identical output after the refactor.
- `uv run pytest tests/test_regression_mvp.py tests/integration/test_mvp_end_to_end.py` — both must pass unchanged.
- `uv run ruff format --check`, `uv run ruff check`, `uv run ty check --exclude tests/` per `AGENTS.md`.

### Stage 2 — Build 4TU #1 loader

**Files to touch:**

- `src/rl_health_interventions/data/loaders.py` — at impl-time decision per Q8: either extend `load_4tu_step_goals` to read the 3.6 MB zip from dataset `53f2d238`, or add `load_4tu_collaborative_goals` alongside it. Whichever preserves more information.

**Expected output schema (per Stage 0 inspection):**

`user_id`, `timestamp`, `prior_day_steps`, `mood`, `sleep_quality`, `available_time`, `motivation`, `self_efficacy`, `assigned_goal_offset` (one of −400, −200, 0, +200, +400), `next_day_steps` (or equivalent next-session steps).

**Verification:**

- Round-trip: loader → polars DataFrame → assert ≥700 transitions, all expected columns present, no nulls in critical columns (`user_id`, `timestamp`, `assigned_goal_offset`, `next_day_steps`).
- New `tests/unit/data/test_load_4tu_collaborative.py` (or extend existing) — assert schema, row count, action-value distribution covers all 5 levels.
- Loader test marked `@pytest.mark.integration` (per `pyproject.toml:40-42`).

### Stage 3 — Feature pipeline (replace stub)

**Files to touch:**

- `src/rl_health_interventions/data/feature_pipeline.py` — replace the stub `from_config` classmethod with a real `from_dataset` classmethod that maps raw loader columns → `StateView.features` dict. Standardise feature naming across datasets where possible.

**Verification:**

- Unit test: feed a small synthetic dataset, assert `FeaturePipeline.from_dataset(ds).to_features(user_id, t)` produces a dict with expected keys.

### Stage 4 — `LearnedTransition` (ridge, residual)

**New file:** `src/rl_health_interventions/transitions/learned.py`

- `class LearnedTransition(TransitionModel)`:
  - `__init__(self, weights: np.ndarray, bias: np.ndarray, feature_names: list[str], action_names: list[str])` — load fitted weights.
  - `transition(state: StateView, action: str) -> StateView` — predict `Δs` per feature, return `dataclasses.replace(state, features={k: (state.features or {}).get(k, 0.0) + delta[k] for k in feature_names})`.
- `register()` function committing `"learned"` key into the `transitions/__init__.py` registry.

**Fit-time script:** `scripts/fit_learned_transition.py` (or `notebooks/fit_learned.ipynb`). Closed-form ridge:

```
design matrix X = [features one-hot(action)] + intercept column
y = s_{t+1} - s_t    (residual target, per Q2)
w = (Xᵀ X + λ I')⁻¹ Xᵀ y    via np.linalg.solve
  where I' is the identity with I'[0,0] = 0 (intercept unpenalised)
λ grid search on LOUO validation split
```

Saves fitted weights + feature names + action names to `data/learned_<dataset>.npz`.

**Verification:**

- `tests/unit/transitions/test_learned.py`:
  - Synthetic known-linear ground truth: assert coefficients recovered within 5%.
  - Residual prediction on `Δs = 0` ground truth: assert near-zero weights.

### Stage 5 — Falsification harness

**New file:** `tests/evaluation/test_ridge_falsification.py`

Three gates, all must pass for ridge to be accepted:

1. **No-op baseline comparison (Q3 + Q5):**
   - Compute `noop_rmse = sqrt(mean(Δs²))` (i.e., `Δŝ = 0` prediction).
   - Compute `ridge_rmse_louo` via leave-one-user-out CV, `ridge_rmse_temporal` via last-20%-of-timeline per-user holdout.
   - Pass criterion: `min(ridge_rmse_louo, ridge_rmse_temporal) ≤ 0.80 × noop_rmse` on at least one primary feature (≥20% improvement, per Q5).
2. **Coefficient sanity — autocorrelation (Q7 + Q9):**
   - Refit a *separate* direct model `s_{t+1} = Ridge(s, a)` (not residual).
   - Assert `w[prior_steps → next_steps] > 0`. Where HR is present, assert `w[prior_hr → next_hr] > 0`.
3. **Output:** JSON report `{dataset, feature, noop_rmse, ridge_rmse_louo, ridge_rmse_temporal, ratio, direct_coef_sign_pass}` saved to `reports/learned_falsification_<dataset>.json`.

**Decision gate:**

- Pass → ridge is the v1 learned transition. Fill in `docs/learned/learned_specification.tex` with the result and proceed to Stage 6.
- Fail → upgrade to NN (Stage 7).

### Stage 6 — Docs (only on Stage 5 pass)

- Replace the placeholder in `docs/learned/learned_specification.tex` with: dataset used, model class, fit procedure, falsification outcome.
- Update `docs/decisions/initial_design.tex` MDP decision log table — mark transition row as `Confirmed: learned (residual ridge) on 4TU#1`.
- Update `docs/overview/ROADMAP.md` to acknowledge the Phase 1 / Phase 2 boundary has been crossed (per grilling Q3 framing). Add a Phase 2 item for the NN upgrade path.

### Stage 7 — NN upgrade (CONDITIONAL, only on Stage 5 fail)

- Add `torch` as a runtime dep in `pyproject.toml`. This violates `AGENTS.md` ("Do not add runtime dependencies unless they are required") and `initial_design.tex` ("Dependencies? Minimal: no Gymnasium, no SB1"). Requires explicit user sign-off at the time.
- **New file:** `src/rl_health_interventions/transitions/learned_nn.py` — `class LearnedNNTransition(TransitionModel)`: `nn.Linear(s_dim + a_dim → hidden → Δs)`, residual output, trained on `(s, a, s_{t+1} - s_t)`.
- Re-run Stage 5 falsification harness on the NN. Document comparison in `docs/learned/learned_specification.tex`.

---

## Dependency discipline

| Stage | New runtime deps | Notes |
|---|---|---|
| 1 (ABC widening) | None | `numpy` already in `pyproject.toml:14`. |
| 2 (loader) | None | `polars` already in `pyproject.toml:16`. |
| 3 (feature pipeline) | None | numpy + polars. |
| 4 (ridge) | None | Closed-form via `np.linalg.solve` (Q6). |
| 5 (falsification) | None | numpy + existing `pytest` infra. |
| 6 (docs) | None | LaTeX only. |
| 7 (NN, conditional) | `torch` | Requires explicit user sign-off. Violates minimal-deps rule. |

All fit-time scripts use `logging` (stdlib), never `print()` (per `AGENTS.md`).

---

## Follow-ups flagged (out of scope for this PR)

These TILES-related doc corrections are *not* done in this PR — they're larger-scope rewrites of validation strategies, not single-section fixes. Tracked here so they don't get lost.

1. **`docs/archive/research/simulator-validation.md`** — 12 references to TILES as an intervention dataset, embedded in validation strategies B (Real-Data Replay), D (Predictive Checks), and the cross-dataset transportability plan. Needs a wholesale rewrite to either (a) drop TILES entirely and rely on HeartSteps V1/V2 alone, or (b) replace TILES with 4TU #1 (the new primary dataset for this plan).
2. **`docs/archive/research/SUPERVISOR_SUMMARY.md:75, 108`** — two sentence-level references that treat TILES as the documented fallback for HeartSteps V2 access delays. Needs rewriting to either drop the fallback or substitute 4TU #1.
3. **`docs/sources/data_availability_schema.md`** — pull TILES into the master table as an observational Fitbit dataset (it's not currently listed at all). Mark `Intervention actions: ❌`.
4. **`docs/decisions/initial_design.tex:90-99`** — design-doc state definition skews physiological (steps, HR, sleep, sedentary min). The chosen 4TU #1 dataset has one physiological feature (prior-day steps) and five psychological. When Stage 1 lands, decide whether to (a) update the design doc state to admit psychological features, or (b) hold out for TILES-equivalent physiological data (which doesn't exist openly).
5. **`docs/overview/ROADMAP.md`** — already explicitly defers `UserProfile`-based transitions, multi-timescale reward, `from_dataset()` to Phase 2 (line 73). This plan crosses that boundary. ROADMAP needs a Phase 2 entry that links to this plan.

---

## Decisions explicitly deferred

- Whether to update `initial_design.tex` state definition (psychological vs physiological) — flagged above, not decided.
- Whether to apply for HeartSteps V1/V2 access (the gold-standard RL-in-the-loop data) — not in scope here.
- NN architecture details if Stage 7 is triggered — gated on Stage 5 outcome.
