> **Archived 2026-06-23.** This document was part of the June-14 research batch
> (PRs #86–#90). It is retained for historical reference but is no longer actively
> maintained. See `docs/research/README.md` for current research artifacts.

# Simulator Validation Strategy — Decision Tree

**Date:** 2026-06-14  
**Author:** Research subagent (Hermes)  
**Status:** Draft for review  
**Scope:** Phase 1–2 simulator validation for the config-driven RL health intervention framework  
**Target paper format:** Nature Methods

> **Known-incorrect section (2026-06-17):** This document references TILES-2018 as a source of "intervention delivery logs" and as a cross-dataset transportability target (Strategies B and D, gap analysis, follow-ups). That characterisation is wrong: TILES-2018 is observational (Mundnich et al. 2020, *Sci. Data* 7:354) and has no `a_t` variable. See `docs/sources/additional_data_sources.md` (Dataset 9, correction note) and `docs/archive/plans/learned_transitions.md` (TILES — DROPPED) for the canonical correction. This document needs a wholesale rewrite to either (a) drop TILES and rely on HeartSteps V1/V2 alone, or (b) substitute 4TU #1 as the open-access intervention validation set. Tracked as a follow-up; not fixed in the same PR as the correction to `additional_data_sources.md`.

---

## Table of Contents

1. [The Problem: Why Validate?](#1-the-problem-why-validate)
2. [The Decision Tree: Six Candidate Strategies](#2-the-decision-tree-six-candidate-strategies)
3. [Strategy Evaluation Matrix](#3-strategy-evaluation-matrix)
4. [Recommendation: Two-Stage Validation](#4-recommendation-two-stage-validation)
5. [Minimum Viable vs. Gold Standard](#5-minimum-viable-vs-gold-standard)
6. [Paper Placement: Main, Supplementary, Future Work](#6-paper-placement-main-supplementary-future-work)
7. [Interaction with the Analysis Plan](#7-interaction-with-the-analysis-plan)
8. [Unresolved Open Questions](#8-unresolved-open-questions)

---

## 1. The Problem: Why Validate?

The project uses a config-driven user simulator with four behavioural archetypes (goal-driven, social responder, resistant, stable maintainer). Phase 1 experiments run entirely in this simulated environment. Phase 2 will replace synthetic profiles with real-data calibrations from HeartSteps V2, All of Us, and UK Biobank.

**The core question:** How do we know the simulator is "good enough" that conclusions drawn from RL experiments inside it transfer to a real NUS cohort study?

Three risks make this non-trivial:

- **Distribution shift.** Synthetic step-count distributions may not match real wearable data, especially for tail behaviours (e.g., highly sedentary participants, weekend patterns).
- **Response model misspecification.** The rule-based response model (delta-steps as a linear function of archetype parameters, state, and action) may not capture real user behaviour — especially burden dynamics, non-stationarity, and context-dependent responses.
- **Confirmation bias.** If the simulator is tuned to produce results the researcher expects, the evaluation is circular.

The validation strategy must therefore answer: *does the simulator produce behaviour that is indistinguishable from real users in the ways that matter for RL policy evaluation?*

---

## 2. The Decision Tree: Six Candidate Strategies

### A. Behavioural Fingerprint Matching
*Compare aggregate step-count distributions between simulated and real cohorts.*

- **What it requires:** ~100+ real participants with continuous wearable data (steps, HR, sleep, sedentary time). No intervention response data needed. Available from: HeartSteps (if shared), All of Us (cloud workbench), NHANES (14.7K participants, CC0, downloadable now).
- **What it proves:** The simulator reproduces the *marginal distribution* of daily steps — mean, variance, autocorrelation, day-of-week effects, burst structure. This is necessary but not sufficient.
- **What it does NOT prove:** That the *conditional response* to interventions is realistic. Two simulators with identical step-count marginals could have radically different action-to-response mappings, and RL policies would behave differently in each.
- **Failure modes:**
  - Matching means and variances while missing temporal structure (bursts, weekend dips, post-intervention recovery patterns).
  - Overfitting: tuning synthetic parameters to match one dataset's fingerprints while failing on a held-out dataset.
  - Aggregate fingerprints mask subgroup heterogeneity. The 4 archetypes could produce population-level fingerprints that match real data even if individual archetypes are wrong (Simpson's paradox analogue).

### B. Real-Data Replay
*Train a known RL algorithm on real data, replay its decisions in the simulator, check that the simulator produces similar state trajectories.*

- **What it requires:** Real MRT data with recorded actions *and* subsequent state transitions (i.e., action-to-response mappings). This means HeartSteps V1/V2 data (requires data sharing agreement; weeks-to-months lead time). Also possible with TILES (open access, intervention delivery logs).
- **What it proves:** For the *specific trajectories* in the replay set, the simulator produces state transitions that match real observed transitions. This directly tests the conditional response model — the hardest thing to validate.
- **What it does NOT prove:**
  - Coverage of the state-action space: the replay only tests transitions that occurred in the real data. If the RL agent in Phase 1 chooses actions the real policy rarely took (e.g., frequent motivational prompts), the simulator may be wrong in unexplored regions.
  - That the simulator generalises to the NUS cohort, which has different demographics, different wearable devices, and a different intervention context.
- **Failure modes:**
  - If real data is too sparse (HeartSteps V1 has ~50 participants × 42 days × 5 decisions/day ≈ 10,500 transitions), the replay set is small and confidence intervals wide.
  - The replay check can pass (low prediction error) because the simulator is degenerate — e.g., the "no message" action dominates real data, and the simulator trivially copies it.

### C. Regret-Bound Calibration
*Run the same RL algorithm in the simulator and in a small MRT (or held-out real-data bandit evaluation) and compare regret curves and final policies.*

- **What it requires:** A real online experiment or an offline policy evaluation (OPE) estimator that can produce regret-bounds from logged data. HeartSteps V2 already ran Thompson Sampling — we have a real regret baseline. For a new NUS cohort, a small MRT (10–20 participants) would be needed.
- **What it proves:** That the simulator produces the same *relative ordering* of policies as the real world. If Thompson Sampling outperforms a fixed baseline by 15% in the simulator and 12% in the real MRT (within confidence intervals), the simulator is useful for policy comparison.
- **What it does NOT prove:**
  - Absolute performance levels: even if rankings match, the simulator may over/underestimate effect sizes.
  - That the policy learned in the simulator is optimal in the real world. The simulator might miss real-world complexities (e.g., seasonal effects, lifecycle changes) that shift the optimal policy.
- **Failure modes:**
  - Real MRTs are expensive and time-consuming. A 20-participant × 90-day MRT costs ~$50K–$100K and takes 6–12 months.
  - OPE estimators (e.g., importance sampling, doubly robust) have high variance when the logging policy and target policy differ — which they inevitably do when the target policy was learned in a simulator.
  - Regret curves smooth over trajectory-level errors. A simulator that is wrong in important states but right on average could still pass.

### D. Predictive Checks (Bayesian Posterior Predictive p-Values)
*Fit a generative model to real data, simulate from it, check that observed real-data statistics fall within the 95% predictive interval.*

- **What it requires:** A fully Bayesian specification of the user model (priors over archetype parameters, response magnitudes, burden dynamics). Real data to condition the posterior. MCMC or variational inference to draw posterior samples.
- **What it proves:** That the simulator generates data consistent with real observations under the chosen model family. Provides a principled (Bayesian) measure of "surprise" — if the real data falls in the tail of the predictive distribution, the model is misspecified.
- **What it does NOT prove:**
  - That the model family is correct. A poor model can still pass predictive checks if the checks are insufficiently sensitive (the "all models are wrong, some are useful" problem).
  - That the simulator is useful for RL policy evaluation. Predictive checks test the data-generating process, not the policy evaluation downstream task.
- **Failure modes:**
  - Computational cost: full Bayesian inference over 4 archetypes + burden dynamics + response models is non-trivial. Could take weeks to implement and debug.
  - Choice of test statistic is critical and somewhat arbitrary. Pick the right statistic and the model passes; pick a different one and it fails. The Bayesian literature acknowledges this as the "double-use of data" problem unless the test statistic is chosen before seeing the data.

### E. Cross-Dataset Transportability
*Calibrate the simulator on dataset A, predict held-out data from dataset B (different population, device, or time period), and measure prediction degradation.*

- **What it requires:** Two (or more) sufficiently rich datasets. The best pairing is: calibrate on HeartSteps V2 (~97 participants, Fitbit, intervention response), predict on TILES (open access, Fitbit, intervention delivery logs, 12 months) or on held-out HeartSteps V1 data. A second pair: calibrate on NHANES step distributions (14.7K participants, ActiGraph), predict on All of Us (59K participants, Fitbit) — tests device generalisation but lacks intervention response.
- **What it proves:** How much the simulator degrades when applied to a different population or measurement tool. If the degradation is small, the simulator is robust and the calibration isn't overfitted to dataset-specific artefacts.
- **What it does NOT prove:**
  - That the simulation is correct — only that the error is consistent across datasets. Two datasets could share the same bias.
  - Transportability to the *actual NUS cohort* unless the NUS cohort is demographically and behaviourally similar to the calibration datasets.
- **Failure modes:**
  - Degradation can be hard to interpret. If the simulator performs worse on dataset B, is it because of population differences, device differences, or study protocol differences? Partitioning these requires careful experiment design (matched cohorts, same device model, same intervention protocol).
  - Requires *multiple* datasets with intervention response data, which are extremely rare. HeartSteps and TILES are the only viable candidates. If TILES doesn't provide fine-grained enough data, this strategy collapses.

### F. Adversarial / Mechanistic Plausibility Checks (Supplement)
*Expert review of behavioural plausibility, stress-testing at extreme parameter values, and sanity checks on mechanistic predictions.*

- **What it requires:** Domain expert time (2–4 hours), no data access. A protocol for stress-testing: set parameters to extremes, check that the simulator produces pathological but interpretable behaviour (e.g., resistant archetype + very high burden penalty → zero response → flat steps).
- **What it proves:** That the simulator is *not obviously wrong* — no impossible transitions, no contradictory incentive structures, no monotonicity violations (e.g., heavier interventions producing *less* cumulative activity in all archetypes).
- **What it does NOT prove:**
  - Any quantitative claim about real-world fidelity. This is a logical check, not an empirical one.
- **Failure modes:**
  - False sense of security: the simulator passes all sanity checks but is still quantitatively wrong.
  - Expert disagreement: two behavioural scientists may disagree on what "plausible" means for burden dynamics.

---

## 3. Strategy Evaluation Matrix

### Comparison Table

| Criterion | A. Fingerprint | B. Data Replay | C. Regret Calibration | D. Predictive Checks | E. Transportability | F. Adversarial |
|-----------|---------------|---------------|----------------------|---------------------|--------------------|---------------|
| **Tests conditional responses?** | No | Yes | Indirect (via policy) | Yes (full joint dist.) | Partial | No |
| **Tests marginal distributions?** | Yes | Yes | No | Yes | Yes | No |
| **Data requirement** | 100+ participants, passive only | MRT data (actions + responses) | MRT data or OPE | MRT data + Bayesian model | 2+ datasets | None |
| **Data availability for Phase 1** | NHANES now | Requires HeartSteps/TILES | Requires HeartSteps V2 | Requires HeartSteps V2 | NHANES + All of Us if accessible | N/A |
| **Compute cost** | Low (summary stats) | Medium (trajectory matching) | Medium-High (full RL runs) | High (MCMC/VI) | Low-Medium | Negligible |
| **Expertise required** | General data science | RL + causal inference | RL + experimental design | Bayesian statistics | Data science | Domain expertise |
| **Time to first result** | 1–2 weeks | 4–8 weeks (data access) | 8–16 weeks (data + coding) | 4–12 weeks (model + inference) | 4–12 weeks | 1 week |
| **Proof strength** | Weak (necessary) | Strong (conditional) | Strong (policy-relevant) | Strong (principled) | Strong (generality) | Weak (sanity only) |
| **Main failure risk** | Misspecification of conditionals | Sparse real data | MRT cost; OPE variance | Test statistic choice | Hard to interpret degradation | False security |

### Feasibility vs. Informativeness Plot

```
High info
  ▲
  │                    D (Predictive checks)*
  │         B (Data replay)   C (Regret calibration)*
  │          E (Transportability)*
  │
  │                     A (Fingerprint)
  │        F (Adversarial)
  │
  └───────────────────────────────────────────► Feasibility
     Easy (Phase 1)                 Hard (Phase 2+)

  * Requires real MRT/intervention data — Phase 2 only.
```

---

## 4. Recommendation: Two-Stage Validation

Given the project's constraints (Phase 1 has synthetic data only; HeartSteps V2 is the closest analogue; no real MRT data until Phase 2; NUS cohort is the ultimate target), I recommend a **three-strategy combination staged across Phase 1 and Phase 2**.

### Stage 1 (Phase 1 — now): Behavioural Fingerprint + Adversarial Checks

**Rationale:** With no real intervention data available in Phase 1, we cannot validate conditional responses. We can, however, ensure the synthetic marginal distributions are realistic and the simulator is internally consistent.

**What to do:**

1. **Parameterise synthetic generators from NHANES** (14.7K participants, CC0, available now). Fit step count distributions stratified by age, sex, and day-of-week. Validate that synthetic steps match NHANES means, variances, autocorrelations (lag-1), and weekend/weekday differences within 5% relative error.

2. **Validate HR-activity-sleep correlations** using MMASH (open access, ODbL). Check that the synthetic multivariate distribution (steps, HR, sleep hours, sedentary minutes) preserves observed correlation structure.

3. **Adversarial stress tests:**
   - Set all action penalties to zero → all archetypes should show the *same* maximal response (the intervention effect is purely beneficial).
   - Set burden decay rate to zero → response should remain suppressed (burden persists, no recovery to baseline).
   - Set all archetypes to the same parameters → the simulator should produce homogeneous behaviour indistinguishable from a single-group model.
   - Remove the "no message" action → burden should saturate all users; the step response should collapse.

**Go/no-go gate:** If synthetic fingerprints deviate >10% from NHANES population statistics on any key dimension (mean steps, variance, weekend dip, autocorrelation), pause and retune the synthetic generator before proceeding to RL experiments.

**What this enables:** The Phase 1 paper can claim "the simulator produces realistic marginal step-count distributions parameterised from the largest public wearable dataset (NHANES, N=14,693)." This is a defensible starting point for *methodology* papers.

### Stage 2 (Phase 2 — post-HeartSteps access): Data Replay + Transportability

**Rationale:** Once HeartSteps V2 data is available (via data sharing agreement), the priority shifts to validating conditional responses — the hardest and most critical aspect.

**What to do:**

1. **Primary: Real-data replay.** Take the HeartSteps V2 logged trajectories (actions selected by the Thompson Sampling bandit, subsequent step-count outcomes), and replay those actions through the simulator. Compute:
   - **Mean squared error (MSE)** of predicted vs. actual delta-steps
   - **Expected calibration error (ECE)** for binned predictions
   - **Coverage** of 80% prediction intervals on held-out users (leave-one-user-out cross-validation)
   - **Ranking preservation**: does the simulator correctly rank which actions are most effective for each archetype?

2. **Secondary: Cross-dataset transportability.** Calibrate the simulator on HeartSteps V2, then test on TILES (open access, intervention delivery logs, 12-month coverage). Measure degradation in prediction MSE. If degradation <30%, the simulator is transportable.

3. **Tertiary: Regret-bound calibration (if budget permits).** Run the same Thompson Sampling implementation from Phase 1 on the HeartSteps V2 logged data (off-policy evaluation) and compare the regret profile to the Phase 1 synthetic results. If the relative ordering of policies is preserved, this is strong evidence that the simulator supports valid RL evaluation.

**Go/no-go gate:** If the replay MSE on HeartSteps V2 exceeds the phase-1 within-simulator prediction error by more than 2×, the response model needs recalibration before the NUS cohort analyses begin.

### Why Not the Others

- **Predictive checks (D):** Too much Bayesian infrastructure for a project that doesn't already use Bayesian methods in its core pipeline. The marginal benefit over data replay is small — replay directly tests the prediction task we care about (action-to-response). Add if a student/postdoc already has Bayesian workflow experience; otherwise, defer.
- **Regret calibration as primary (C):** Logistically prohibitive in Phase 1 (no real MRT), and even in Phase 2, one would need to run a new MRT or have high-quality OPE. Use as a confirmatory check in Stage 2 if resources permit.
- **Adversarial as primary (F):** Necessary but insufficient. Include as a sanity gate in Stage 1 but don't stop there.

---

## 5. Minimum Viable vs. Gold Standard

### Minimum Viable Validation (for Phase 1 publication defensibility)

| Component | What | Metric | Pass threshold |
|-----------|------|--------|----------------|
| Marginal step distributions | NHANES parameterised synthetic data | Mean, SD, autocorrelation, weekend effect | All within 10% of NHANES |
| Archetype distinctiveness | ANOVA on step distributions | F-statistic | p < 0.01 of the null hypothesis |
| Adversarial checks | Stress tests from Stage 1 | All pass | 0 behavioural contradictions |
| Real-data replay | HeartSteps V2 (if shared) or TILES | MSE of delta-steps | Within 2× of within-simulator error |
| Paper claim | "Simulator produces realistic marginal distributions from NHANES; conditional responses validated against HeartSteps V2" | — | — |

### Gold Standard Validation (ideal for Nature Methods)

| Component | What | Metric | Pass threshold |
|-----------|------|--------|----------------|
| All of the above | — | — | — |
| Cross-dataset transportability | Calibrate on HeartSteps V2, test on TILES | MSE degradation | <30% increase |
| Regret-bound calibration | Compare Phase 1 regret profiles with real-data OPE regret | Kendall rank correlation | τ > 0.8 |
| Policy ranking preservation | Bootstrap CI on ranking | Probability correct ordering | ≥90% |
| Ablation: effect of model mis-specification | Artificially corrupt simulator parameters and check that the validation detects the corruption | Sensitivity | ≥80% of moderate corruptions detected |
| Paper claim | "Simulator produces realistic conditionals (validated via data replay and cross-dataset transportability). Policy rankings are preserved (τ > 0.8). Model misspecification is detectable (80% sensitivity)." | — | — |

---

## 6. Paper Placement: Main, Supplementary, Future Work

### Main Paper
- **Validation philosophy** (1 paragraph in Methods): State the two-stage approach, justify why behavioural fingerprint + data replay is sufficient.
- **Marginal validation results** (1 figure + 1 table in Results): Simulated vs. NHANES step distributions (mean, variance, autocorrelation, weekday/weekend). Mention N=14,693.
- **Real-data replay results** (1 figure in Results): Predicted vs. actual delta-steps for HeartSteps V2 (or TILES). MSE, calibration curve, per-archetype breakdown.

### Supplementary Information
- Full adversarial stress test results (2–3 pages).
- Cross-dataset transportability analysis (calibrate on HeartSteps V2, predict on TILES).
- Sensitivity analysis: how validation metrics vary with simulation parameters.
- Bayesian predictive checks if implemented (can go here even if not in main paper).
- Dataset characteristics tables (NHANES, HeartSteps V2, TILES).

### Future Work
- Regret-bound calibration in a prospective MRT (requires new funding and ethics approval).
- Full Bayesian posterior predictive checks with model comparison.
- Online validation: once the NUS cohort study starts, compare ex-ante simulator predictions with ex-post observed outcomes. This is the strongest validation and should be a dedicated follow-up paper.
- LLM-based user simulation validation (stretch goal — separate decision tree needed).

---

## 7. Interaction with the Analysis Plan

The validation strategy directly shapes several analysis-plan decisions:

### Sample Size Determination
- **For within-simulator RL comparisons:** The analysis plan (M-08) needs to specify a minimum detectable effect size (MDES). The validation strategy determines the *credibility* of that MDES. If the simulator's prediction error on real data is ±20%, then any claimed improvement below 20% is within the noise floor. The analysis plan should therefore:
  - Report the real-data validation MSE alongside the within-simulator comparison.
  - Pre-register a *noise-adjusted* MDES: `max(simulator_MDES, real_validation_MSE × 2)`.
- **For cross-validation splits with real data:** The transportability analysis determines how many datasets are needed. If degradation is high (>30%), you may need a third dataset (e.g., All of Us Fitbit) for ensemble calibration, which affects sample size calculations.

### Baseline Set
- The validation strategy identifies which baselines are most informative for the paper:
  - **Random and "no message always"** — essential for establishing the lower bound (validated by fingerprint matching).
  - **HeartSteps V2's Thompson Sampling** (if we can approximate it from the logged policy) — the most important external baseline. Comparing our Thompson Sampling implementation against HeartSteps V2's results is a direct real-data check.
  - **Fixed-interval heuristic** (e.g., 3 interventions/day on a schedule) — tests whether the simulator's burden dynamics produce realistic engagement decay.

### Subgroup Analyses
- If data replay reveals that the simulator is accurate for some archetypes but not others (e.g., stable maintainers are well-modelled but resistant users are not), the analysis plan should:
  - **Pre-register per-archetype validation metrics** — report MSE separately for each of the 4 archetypes in a table.
  - **Flag archetypes with MSE > 2× the population mean** — any RL conclusions about these archetypes must be caveated, and sample sizes for these subgroups may need to be larger.
  - **Consider collapsing poorly-modelled archetypes** for the primary analysis. If two archetypes produce indistinguishable step responses in real data, they should be merged for the primary analysis and separated only in secondary analyses.

### Power Analysis
- The validation strategy's noise floor directly feeds the power analysis. If the real-data validation MSE is σ²_val and the within-simulator variance is σ²_sim, the effective variance for detecting a real-world effect of size δ is:
  `σ²_eff = σ²_sim + σ²_val` (assuming independence).
- The analysis plan should compute power at `δ / √(σ²_sim + σ²_val)`, not `δ / σ²_sim`. This is a pre-registration commitment.

### Baseline Balance
- If cross-dataset transportability shows that the simulator systematically overestimates the effect of one action type (e.g., motivational prompts are too effective), the analysis plan should pre-register a **"validation-calibrated correction"**: a scaling factor applied to simulated effect sizes before comparison with HeartSteps V2 benchmarks.

---

## 8. Unresolved Open Questions

1. **HeartSteps V2 data access timeline?** The entire Stage 2 validation strategy depends on it. If HeartSteps data is delayed beyond the Phase 2 deadline, the fallback is to use TILES (open access, intervention delivery logs, 12 months, ~200 participants) as the primary real-data validation set. TILES has no RL-in-the-loop data, so the regret-bound calibration (Strategy C) would be infeasible; Strategy B (data replay) would still work.

2. **What is the acceptable validation noise floor?** A 10% noise floor for step-count prediction is likely acceptable; 30% may not be. The project needs a *pre-specified* threshold pre-registered before any validation analysis begins. I recommend: "The simulator will be deemed valid for RL policy evaluation if (a) marginal step distributions match NHANES within 10%, (b) real-data replay MSE is within 2× of within-simulator error, and (c) no adversarial check fails." This should go in the analysis plan pre-registration.

3. **Should the 4 archetypes be validated as discrete categories or as a continuous latent space?** The current plan uses discrete archetypes. If real data shows continuous variation (which it almost certainly will), the validation strategy needs to decide whether to: (a) keep discrete archetypes and accept reduced fidelity; (b) replace with an embedding/regression-based response model; or (c) keep discrete archetypes for the paper but validate that the continuous-to-discrete mapping preserves the important distinctions (e.g., did the top 10% most-resistant users actually behave like our resistant archetype?). This decision affects the Stage 2 replay analysis.

4. **Accountability: what happens if the simulator fails the Stage 1 gate?** If NHANES fingerprints deviate >10%, the project has two options: (a) recalibrate the synthetic generator (tune parameters until the match is within 10%); or (b) document the deviation transparently as a limitation and run sensitivity analyses to bound its effect on RL policy ordering. Option (b) may be acceptable for a methods paper but not for a clinical recommendations paper. This decision should be pre-specified.

5. **Does the validation strategy require a formal pre-registration?** Yes, if the paper is targeting Nature Methods. Pre-register the two-stage strategy, the pass thresholds, the noise-floor adjustment, and the per-archetype reporting plan before any real-data analysis begins. This prevents accusations of post-hoc validation gate-moving.

---

## Summary Decision Tree

```
START HERE
│
├─ Can we access HeartSteps V2 or TILES data?
│   NO  ──► Stage 1 only: Fingerprint matching (NHANES) + Adversarial checks
│   │            │
│   │            └── Acceptable for methods paper, not for clinical claims
│   │
│   YES ──► Stage 2: Data replay on HeartSteps V2 (primary)
│            │
│            ├── Also test TILES? ──► Transportability check (stronger paper)
│            │
│            ├── OPE feasible? ──► Regret-bound calibration (gold standard)
│            │
│            └── Bayesian infrastructure available? ──► Predictive checks (supplementary)
│
└── Regardless: Adversarial checks (Stage 1, quick win)
```

**Bottom line:** Commit to Stage 1 now (NHANES fingerprints + adversarial checks). Start HeartSteps data access requests immediately. Pre-register the two-stage strategy with explicit pass thresholds. Plan for Stage 2 data replay as the primary validation result. Treat cross-dataset transportability and regret calibration as aspirational extras that strengthen the paper but are not required for publication.

The two-stage strategy is the only feasible path that produces *some* quantitative validation in Phase 1 (fingerprint matching) and *strong* conditional validation in Phase 2 (data replay) when the scarce HeartSteps data finally arrives. No single strategy from the decision tree provides both feasibility and strong proof; the combination is necessary.
