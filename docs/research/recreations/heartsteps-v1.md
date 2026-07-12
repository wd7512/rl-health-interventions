---
title: "HeartSteps V1 Recreation Report"
status: "active"
last_reviewed: "2026-07-12"
---

# HeartSteps V1 Recreation Report

**Recreation target:** Klasnja et al. (2015, UbiComp/ISWC) — MRT methodology paper; Klasnja et al. (2019, *Annals of Behavioral Medicine*) — efficacy results.

**Date:** 2026-06-15

**Branch:** `research/recreate-heartsteps-v1`

---

## 1. Paper Overview

HeartSteps V1 was a 6-week micro-randomized trial (MRT) conducted to optimize a just-in-time adaptive intervention (JITAI) for increasing physical activity among sedentary adults. Participants wore a Jawbone Up Move activity tracker and carried an Android phone running the HeartSteps application. At five user-specified times each day, the system randomized whether to deliver a contextually tailored activity suggestion — either a walking suggestion or an anti-sedentary suggestion — or no suggestion. Suggestions were tailored on time of day, day of week, location (home/work/other), and weather. The proximal outcome was step count in the 30-minute window following each decision point.

The trial is important for three reasons. First, it introduced MRTs to the mHealth community as a rigorous experimental design for optimizing intervention components before committing to a full efficacy trial. Second, it produced the first empirical evidence that contextually tailored activity suggestions can increase short-term walking bouts (14% increase, p = 0.06). Third, the V1 dataset became the training prior for the HeartSteps V2 RL algorithm (Liao et al. 2019), making V1 the empirical foundation of the RL-in-the-loop mHealth paradigm. This recreation documents what the project can and cannot reproduce, connects V1 findings to the current framework, and identifies concrete next actions.

---

## 2. Headline Numbers

All numbers below are drawn from the open-access PMC version of Klasnja et al. (2019), the Qian et al. (2020) tutorial paper (arXiv:2004.10241), and the Yu & Qian (2025) follow-up analysis (arXiv:2410.15049). Section numbers refer to Klasnja et al. (2019) unless noted.

| Metric | Value | Source |
|---|---|---|
| Analyzed participants | 37 (from 44 enrolled) | Results, CONSORT diagram (Fig. 2) |
| Trial duration | 42 days (6 weeks) | Methods, Randomization |
| Decision times per day | 5 (user-configured) | Methods, Randomization |
| Max decision points per participant | 210 (42 × 5) | Methods, Randomization |
| Total generated decision points | 8,274 | Results, Included Decision Points |
| Included after exclusions | 7,540 | Results, Included Decision Points |
| Available decision points (analyzed) | 6,061 | Results, Available Decision Points |
| Theoretical max decision points | 7,770 (37 × 210) | Derived |
| Availability rate | ~80% (19.6% unavailable) | Results, Available Decision Points |
| Randomization (any suggestion vs none) | 0.6 / 0.4 | Methods, Randomization |
| Randomization (walking / anti-sedentary / none) | 0.3 / 0.3 / 0.4 | Methods, Randomization |
| Mean 30-min step count (overall) | 253 (SD = 473) | Results, Effects of Activity Suggestions |
| ATE (any suggestion) — exponentiated | eb = 1.14 (p = 0.06) | Results, Effects of Activity Suggestions |
| ATE (any suggestion) — absolute | +35 steps (95% CI ≈ −2 to +78) | Derived from eb = 1.14 [0.99, 1.31] × 253 |
| ATE (any suggestion) — WCLS log-scale | β = 0.131 (p = 0.06, 95% CI −0.006 to 0.268) | Qian et al. 2020, Table 1 |
| Initial effect (day 1, any suggestion) | eb = 1.66, +167 steps (p < 0.01) | Results, Effects of Activity Suggestions |
| Effect decay per day | −2% (eb = 0.98, p < 0.01) | Results, Effects of Activity Suggestions |
| Effect extinguished by | Day 28 | Results, Effects of Activity Suggestions |
| Walking suggestion ATE | eb = 1.24, +59 steps (p = 0.02) | Results, Walking vs Antisedentary |
| Walking suggestion initial (day 1) | eb = 2.07, +271 steps (p < 0.001) | Results, Walking vs Antisedentary |
| Walking suggestion decay | −3%/day (eb = 0.97, p < 0.001) | Results, Walking vs Antisedentary |
| Anti-sedentary ATE | Not statistically significant | Results, Walking vs Antisedentary |
| Location moderation (walking) | β₁ = 0.377, p < 0.05 | Qian et al. 2020, Section 5, Table 3 |
| Power analysis target | N = 32 for 80% power, effect size 0.1, 70% avail, α = 0.05 | Methods, Statistical Analyses |
| Suggestions delivered | 3,594 (59.3% of available) | Results, Available Decision Points |
| Walking suggestions | 2,015 (33.2%) | Results, Available Decision Points |
| Anti-sedentary | 1,579 (26%) | Results, Available Decision Points |
| Avg suggestions/participant/day | 2.4 (SD = 1.1) | Results, Available Decision Points |

### Numbers NOT verified from the open paper text

The following numbers from the requirements checklist could not be located in the open-access versions of Klasnja et al. (2015, 2019) or their published supplementary materials. They may appear in the supplementary DOCX file (not publicly accessible at time of writing) or result from different model specifications:

| Requested number | What the paper reports instead |
|---|---|
| ATE: +13.9 steps (95% CI −14.9 to 42.7) | Paper: +35 steps (95% CI ≈ −2 to +78, derived from eb = 1.14 [0.99, 1.31] × 253). The narrower range (−14.9 to 42.7) suggests a different model or data subset. |
| Log-scale β = 0.041 (SE = 0.037, p = 0.27) | Paper (Klasnja 2019): β = 0.131 (p = 0.06). Yu & Qian (2025): β = 0.200 (SE = 0.087, p = 0.029) for walking suggestions. The value 0.041 may come from a different outcome window or covariate set not publicly documented. |
| Location: gym +116.7 steps, p < 0.05 | Paper reports location moderation on log-scale (β₁ = 0.377, Qian et al. 2020 Table 3). The specific absolute value +116.7 was not found in open-access text. |
| Power: 80% to detect 60-step increase | Paper reports powering for standardized effect size 0.1 (half of Cohen's small = 0.2), N = 32. Absolute step equivalent not specified. |
| Availability rate ~72% | Paper reports ~80% (19.6% unavailable). The 72% may be a different definition or pre-study projection. |

---

## 3. Recreation Methodology

### What we CAN recreate

1. **Generative model parameters.** The V1 paper and associated analysis code (public on GitHub at `klasnja/HeartStepsV1` and `StatisticalReinforcementLearningLab/HeartstepsV1Code`) document the WCLS estimation procedure, the feature encoding (log pre-treatment step count, location indicators, day-of-week indicators), and the randomization probabilities. These can be re-implemented in Python using our framework.

2. **GEE / WCLS specification.** The centered-and-weighted least-squares estimator (Boruvka et al. 2018) is fully specified in the open-access methods. The R package `MRTAnalysis` provides a `wcls()` function that is a reference implementation. We can re-implement the same estimating equations in our `src/rl_health_interventions/` module.

3. **Proximal outcome model.** The paper defines the proximal outcome as log(30-min step count + 0.5), controlling for log(pre-30-min step count + 0.5). This transformation and analysis model is fully reproducible.

4. **Heterogeneity / moderation analysis.** The location, weekday, and activity-level moderation models are documented in Yu & Qian (2025) and Qian et al. (2020), with R reference code in the `jiaxin4/MobileHealth` repository.

5. **Power analysis.** The MRT sample-size calculator (Liao et al. 2016) is publicly available. We can verify the N = 32 calculation given the stated parameters (effect size 0.1, 70% availability, α = 0.05, 80% power).

### What we CANNOT recreate

1. **Individual-level participant data.** The V1 dataset is available on GitHub at `klasnja/HeartStepsV1` under a data-use agreement requiring IRB approval and a signed confidentiality agreement. We do not currently have access.

2. **Exact prior coefficients.** The HeartSteps V2 prior (Liao et al. 2019, Section 6.1) is computed from V1 data that we cannot access. The exact numerical values of the prior mean and covariance matrix are therefore not reproducible without data access.

3. **Numerical replication of the primary analysis.** The WCLS estimate β = 0.131 (p = 0.06) requires the full V1 dataset to reproduce. We can re-implement the estimator but cannot verify the specific numerical output.

### What we USE instead

- **NHANES minute-level step count data** (PhysioNet, CC0, 14,693 participants) as a proxy for population step distributions. Already integrated in PR #85's `nhanes_loader.py`.
- **Synthetic data** generated from a parametric model calibrated to NHANES distributions and the V1 paper's reported effect sizes.
- **The R package `MRTAnalysis` with `data_mimicHeartSteps`** — a synthetic dataset (7,770 rows, 37 users × 210 time points) provided by the package authors that mimics the V1 structure exactly, including availability flags, log-transformed outcomes, location indicators, and randomization probabilities. This dataset is documented at <https://rdrr.io/cran/MRTAnalysis/man/data_mimicHeartSteps.html>.

---

## 4. The Four-Question Loop

### 4.1 ATE: Any Suggestion vs. No Suggestion

**1. Is this at the theoretical limit?** The marginal ATE of +35 steps (eb = 1.14, p = 0.06) is below conventional significance thresholds, and the CI includes zero. This is likely not purely a power issue (the study was powered at N = 32; they had 37). It may reflect genuine heterogeneity — the effect is real for some people in some contexts and absent for others, making the marginal estimate noisy. The Bonferroni-corrected threshold (α = 0.05 / 2 primary analyses = 0.025) is not survived by the marginal ATE. The effect is scientifically meaningful (14% increase in walking bouts) but statistically marginal.

**2. How does this link to other papers?** The V1 marginal ATE is the foundation for the V2 RL algorithm (Liao et al. 2019). The V2 paper uses the V1 data to construct the prior for the Bayesian regression, effectively treating the V1 marginal effect as a starting point for personalization. Liao et al. report that 29/37 participants (78.4%) show improvement over a TS Bandit baseline, suggesting the V1 data do contain a treatment signal that personalization can exploit. Yu & Qian (2025) later show that the marginal effect masks a strong early-window effect (minutes 10–25 post-suggestion) that is diluted by the 30-minute aggregate.

**3. Is there still room for improvement?** Yes. The marginal ATE is a coarse summary. The V1 data have not been fully exploited for (a) time-varying effect estimation (Yu & Qian 2025 show the effect peaks at ~20 minutes), (b) individual-level heterogeneity (only 37 participants limit this), and (c) interaction with burden/engagement decay. Our framework's multi-timescale reward model could formalize these decay dynamics.

**4. Take action.** Reproduce the WCLS marginal analysis on `data_mimicHeartSteps` from `MRTAnalysis` to validate our WCLS implementation. Compare our Python WCLS estimate to the R reference (expected β ≈ 0.157, 95% CI 0.031–0.284, per the `wcls()` vignette). This establishes that our estimation pipeline is correct before attempting any proxy-data analysis.

### 4.2 Effect Heterogeneity by Location

**1. Is this at the theoretical limit?** The location moderation analysis (Qian et al. 2020, Table 3) shows a significant interaction (β₁ = 0.377, p < 0.05) for walking suggestions when home/work vs. other. The number of contextual features is small (location, weekday, recent activity), and the interaction model is linear. A more flexible moderation model could find richer patterns.

**2. How does this link to other papers?** Location moderation in V1 directly motivates the context-aware state space in V2 (Liao et al. 2019, Section 5.1 uses location as a feature in f(S_t)). It also connects to the broader JITAI literature (Nahum-Shani et al. 2018). Our framework's state space already includes location-type features in the MDP formalisation (`initial_design.tex`).

**3. Is there still room for improvement?** Yes. V1 only had coarse location categories. Modern smartphone OSes provide finer-grained data (GPS, geofences, POI proximity). Yu & Qian (2025) show that location moderation interacts with the minute-level response curve — the effect of being home vs. work changes the *timing* of the walking response, not just its magnitude. Our framework should model this as a feature × time interaction.

**4. Take action.** Add the V1 location moderation finding as a validation target for the user simulation module (Subphase 1C). The `RuleBasedResponse` should produce larger treatment effects when the simulated user is at "home" or "work" vs. "other," consistent with the V1 moderation estimate β₁ = 0.377. Add a unit test asserting this directional constraint.

### 4.3 Availability Rate

**1. Is this at the theoretical limit?** The observed availability rate of ~80% is consistent with the pre-study projection of 70% used for power calculations. The theoretical maximum is set by the availability definition (unavailable while walking/running, 90 s post-activity, while driving, or offline). These are safety- and burden-driven, so ~80% is likely near the ceiling for this design.

**2. How does this link to other papers?** The V1 availability definition was inherited by V2 (Liao et al. 2019, Section 6.1 assumes p_avail ≈ 0.85). The `data_mimicHeartSteps` package uses a similar mechanism. Our PR #85 simulation currently uses p_avail = 0.85, which aligns with V2 but is above V1's observed rate. The burden dynamics in our framework should interact with availability.

**3. Is there still room for improvement?** Yes. The V1 availability model is binary and rule-based. Modern approaches could use predictive models of receptivity (e.g., from phone usage, location entropy, or prior engagement). Yu & Qian (2025) use the same binary indicator without re-estimation.

**4. Take action.** Adjust the framework's default availability probability to 0.80 (matching observed V1 rate) instead of the current 0.85 (V2 generative model). Add availability as a configurable parameter in `DataConfig` with documented defaults from both V1 and V2.

### 4.4 Power Analysis

**1. Is this at the theoretical limit?** The V1 power analysis (Liao et al. 2016) targets 80% power for effect size 0.1 with 70% availability and α = 0.05, yielding N = 32. The study enrolled 37, exceeding the minimum. This is not at the theoretical limit — the MRT calculator assumes a particular working correlation and constant effects. With small-sample correction (Boruvka et al. 2018), effective power may be lower.

**2. How does this link to other papers?** The V1 power analysis directly informed V2 sample size (97 participants, 90 days). It is also the basis of the MRT sample-size R package (`MRTSampleSize`). Our project's Statistical Analysis Plan (PR #86) should reference V1's power analysis as justification for our planned N = 200.

**3. Is there still room for improvement?** Yes. The V1 power calculation was conservative (half of Cohen's small = 0.1). The observed effect size in V1 (β = 0.131, d ≈ 0.07) is smaller. Our planned N = 200 should be validated against the V1 effect size using the same MRT calculator. If the walking-suggestion-only effect (β = 0.200, d ≈ 0.11) is used as the target, N = 200 may be sufficient; for the marginal effect it may not be.

**4. Take action.** Run the MRT sample-size calculator with the V1-observed effect sizes to determine N required for 80% power. Document this in the Statistical Analysis Plan (PR #86). If N > 200, flag that our planned N may be underpowered for a marginal ATE and consider walking-suggestion-only as the primary estimand.

---

## 5. What This Recreation Validates

**Honest assessment.** This recreation validates the *procedural pipeline* but not the *numerical results*. Specifically:

**Validated.** The project's algorithm pipeline (Thompson Sampling, Bayesian regression with action-centering, dosage variable, proxy value function) faithfully implements the equations in Liao et al. (2019), which in turn build on the V1 WCLS framework. The `paper_reproduction/` code (PR #85) has 112 passing tests confirming key mathematical operations match the paper. The WCLS estimation procedure is understood and could be re-implemented.

**Not validated.** The numerical values of the V1 treatment effects (β = 0.131, eb = 1.14) cannot be reproduced because the V1 dataset requires IRB approval and a data-use agreement the project does not currently have. The NHANES proxy data (PR #85) does not contain an intervention signal, so synthetic-data analyses cannot substitute for real V1 findings. The `data_mimicHeartSteps` synthetic dataset can validate the *implementation* of the WCLS estimator but not the *inferences* drawn from real participants.

**Bottom line.** This recreation is a methodology check, not a replication. It tells us that if we had the V1 data, our code could process it correctly. It does not tell us whether the V1 findings would hold in a new sample or in our framework.

---

## 6. Open Questions

### Q1: Should we attempt a GEE/WCLS on NHANES data as a proxy for the V1 prior?

NHANES contains step-count data but no intervention delivery. A WCLS fit on NHANES would estimate the null distribution (no treatment effect expected), which could serve as a sanity check for the software implementation. It cannot substitute for the V1 prior. Decision: useful but lower priority than securing V1 data access.

### Q2: Should we add V1's heterogeneity features to the framework's state space?

The V1 location moderation finding (β₁ = 0.377 for home/work vs. other) is a concrete quantitative target for the state-space design. Our `StateView` should include at minimum the same three location categories (home / work / other) used in V1. The current design uses continuous coordinate features rather than categorical. Decision: add categorical location encoding before Phase 2 validation.

### Q3: Do our burden dynamics match the V1 finding that effect diminishes with prior activity?

The V1 effect decay (2% per day) is attributed to habituation, not burden per se. Our burden model (linear accumulator with threshold) captures *over-delivery* burden but not *novelty* decay. Even well-timed suggestions lose effectiveness with repetition. Our framework may need a separate "novelty" or "surprise" mechanism. Decision: open for the user simulation redesign in Subphase 1C.

---

## 7. Citation Block

### Klasnja et al. 2015 (MRT methodology, UbiComp/ISWC)

```bibtex
@inproceedings{klasnja2015heartsteps,
  author    = {Klasnja, Predrag and Hekler, Eric B. and Shiffman, Saul and
               Boruvka, Audrey and Almirall, Daniel and Tewari, Ambuj and
               Murphy, Susan A.},
  title     = {Microrandomized trials: An experimental design for developing
               just-in-time adaptive interventions},
  booktitle = {Proceedings of the 2015 ACM International Joint Conference on
               Pervasive and Ubiquitous Computing (UbiComp)},
  year      = {2015},
  pages     = {909--920},
  publisher = {ACM},
  doi       = {10.1145/2750858.2805846}
}
```

### Klasnja et al. 2019 (efficacy results, Annals of Behavioral Medicine)

```bibtex
@article{klasnja2019heartsteps,
  author  = {Klasnja, Predrag and Smith, Shawna and Seewald, Nicholas J. and
             Lee, Andy and Hall, Kelly and Luers, Brook and Hekler, Eric B. and
             Murphy, Susan A.},
  title   = {Efficacy of contextually tailored suggestions for physical
             activity: A micro-randomized optimization trial of {HeartSteps}},
  journal = {Annals of Behavioral Medicine},
  volume  = {53},
  number  = {6},
  pages   = {573--582},
  year    = {2019},
  doi     = {10.1093/abm/kay067}
}
```

### Key supporting references

```bibtex
@article{liao2019heartsteps,
  author  = {Liao, Peng and Greenewald, Kristjan and Klasnja, Predrag and
             Murphy, Susan A.},
  title   = {Personalized {HeartSteps}: A reinforcement learning algorithm for
             optimizing physical activity},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and
             Ubiquitous Technologies},
  volume  = {4},
  number  = {1},
  pages   = {1--22},
  year    = {2020},
  doi     = {10.1145/3381007}
}

@article{boruvka2018assessing,
  author  = {Boruvka, Audrey and Almirall, Daniel and Witkiewitz, Katie and
             Murphy, Susan A.},
  title   = {Assessing time-varying causal effect moderation in mobile health},
  journal = {Journal of the American Statistical Association},
  volume  = {113},
  number  = {523},
  pages   = {1112--1121},
  year    = {2018},
  doi     = {10.1080/01621459.2017.1305274}
}

@article{qian2020methods,
  author  = {Qian, Tianchen and Walton, Ashley E. and Collins, Linda M. and
             Klasnja, Predrag and Lanza, Stephanie T. and Nahum-Shani, Inbal
             and Rabbi, Mashfiqui and Russell, Michael A. and Walton, Maureen A.
             and Yoo, Hyesun and Murphy, Susan A.},
  title   = {The micro-randomized trial for developing digital interventions:
             Data analysis methods},
  journal = {arXiv preprint},
  year    = {2020},
  note    = {arXiv:2004.10241}
}

@article{yu2025modeling,
  author  = {Yu, Jiaxin and Qian, Tianchen},
  title   = {Modeling time-varying effects of mobile health interventions using
             longitudinal functional data from {HeartSteps} micro-randomized trial},
  journal = {arXiv preprint},
  year    = {2025},
  note    = {arXiv:2410.15049}
}
```

---

## 8. Next Actions

Priority-ordered:

1. **Reproduce the WCLS marginal analysis on `data_mimicHeartSteps`.** Implement the centered-and-weighted least-squares estimator in Python using the `MRTAnalysis` R package as a reference. Verify that our β estimate matches the expected value (β ≈ 0.157, 95% CI 0.031–0.284, from the `wcls()` vignette). This validates our estimation pipeline without needing real V1 data. *Estimate: 2–3 days.*

2. **Adjust default availability probability to 0.80.** Change the default `p_avail` in the generative model from 0.85 (V2 prior assumption) to 0.80 (V1 observed rate). Add to `DataConfig` as a documented, configurable parameter with both V1 and V2 defaults in comments. *Estimate: 0.5 day.*

3. **Add categorical location encoding to `StateView`.** The current design specifies location as continuous coordinates. Add a categorical location field (home / work / other) to align with the V1 state space and the moderation finding (β₁ = 0.377). Update the user simulation's `RuleBasedResponse` to produce location-dependent treatment effects consistent with V1. *Estimate: 2 days.*

4. **Run the MRT power calculator with V1-observed effect sizes.** Use the Liao et al. (2016) sample-size calculator to compute the required N for 80% power given β = 0.131 (walking-only: β = 0.200). Document the result in the Statistical Analysis Plan (PR #86). If N > 200, flag the underpowering risk. *Estimate: 1 day.*

5. **Submit a data-access request for the HeartSteps V1 dataset.** The data is available at `github.com/klasnja/HeartStepsV1` under a data-use agreement. Prepare the IRB protocol and analysis plan required by the HeartSteps Investigator Team (ClinicalTrials.gov NCT03225521). Without real V1 data, all reproductions remain methodology checks. *Estimate: 4–8 weeks for approval.*
