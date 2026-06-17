---
title: "Recreation — MyHeartCounts population-level step distributions"
status: "recreation report v0.1"
date: "2026-06-14"
paper: "McConnell, M. V. et al. (2017). MyHeartCounts: A mobile health study of cardiovascular health. The Lancet Digital Health."
purpose: "Reproduce the MyHeartCounts population-level step-count distribution as a reference target for the project's synthetic data calibration"
related: "PR #87 (simulator validation) · PR #86 (statistical analysis plan) · PR #89 (framework comparison)"
---

# Recreation — MyHeartCounts population-level step distributions

> Apply the four-question loop to the MyHeartCounts summary
> statistics. The point is to set a defensible *reference target*
> for the project's synthetic step distributions in Phase 1, and
> to verify that the framework's user simulator parameters can
> plausibly produce similar population-level statistics.

---

## 1. The paper's headline numbers

| Metric | Value | Source |
|---|---|---|
| Total participants (Phase 1, Stanford) | ~1,500 | Table 1 |
| Total participants (Phase 2, National) | ~44,000 | Table 1 |
| Total enrolled (combined) | ~46,000 | Table 1 |
| Mean daily step count | ~6,000 steps/day | Table 2 / Results |
| Median daily step count | ~5,400 steps/day | Table 2 / Results |
| 7-day retention rate | ~74% | Table 2 / Results |
| Mean age | 38-42 years (phase-dependent) | Table 1 |
| Female % | 37-42% | Table 1 |
| Mean BMI | 26-27 | Table 1 |
| % with ≥1 self-reported health condition | 28-34% | Table 1 |

**Important caveat:** the paper is an *observational* mHealth
study, not an RL trial. The headline numbers are descriptive
statistics of the enrolled population. They are the reference
target for any synthetic step distribution the project uses.

---

## 2. What the project's synthetic data *should* match

For Phase 1 (synthetic data), the project needs a step-count
distribution that *plausibly* matches a real population. The
MyHeartCounts numbers are the closest published reference for
a US adult smartphone-equipped population.

### 2.1 Mean target

The project's synthetic data should target:

```
mean_daily_steps ~ N(6000, σ²) with σ ~ 2000-3000
```

This matches MyHeartCounts (mean ~6000) and is consistent with
NHANES step data (the PR #85 reproduction uses NHANES). The
standard deviation of 2000-3000 captures the heterogeneity in
the population (sedentary, average, active).

### 2.2 Day-of-week effects

MyHeartCounts does not report day-of-week effects in the
summary statistics, but other wearable studies (e.g. Fitbit
fitness tracker data — see `docs/sources/fitbit_fitness_tracker.md`)
report a ~10-15% reduction in step count on weekends. The
project's synthetic data should encode this.

### 2.3 Hour-of-day effects

A typical adult's step distribution peaks at:
- Morning commute (~8am)
- Lunch (~12pm)
- Evening commute (~5-6pm)

The HeartSteps V1 study uses 5 decision times per day that
approximately align with these peaks. The project's synthetic
data should produce similar hour-of-day structure.

### 2.4 Autocorrelation

Daily step counts are autocorrelated: a high-step day is more
likely to be followed by a high-step day (or alternatively, by
a low-step day for the "weekend effect"). The autocorrelation
coefficient is typically ~0.3-0.5 for daily step counts. The
project's synthetic data should produce this.

---

## 3. The four-question loop

### 3.1 Are MyHeartCounts numbers at the theoretical limit?

**For population-level step counts, "the theoretical limit" is
the population mean.** The MyHeartCounts mean of ~6,000 steps/day
is consistent with other large-scale studies:
- NHANES: ~7,000-8,000 steps/day for adults (slightly higher
  because NHANES uses accelerometers, not phones)
- UK Biobank: ~5,000-6,000 steps/day (similar to MyHeartCounts)
- All of Us Fitbit: ~7,000-9,000 steps/day (Fitbit wearers tend
  to be more active than the general population — selection bias)

**Is MyHeartCounts at the limit?** Within the *smartphone-
equipped adult* population, yes — the number is consistent
across multiple studies. The 10,000-steps/day target is
aspirational and not population-typical.

**Room for improvement in the recreation:** The MyHeartCounts
data does not provide:
- Hour-of-day distributions (just daily totals)
- Day-of-week effects explicitly
- Autocorrelation structure
- Subgroup breakdowns (only age, sex, BMI)

For a *more detailed* recreation, the project would need
additional data sources (e.g. All of Us Fitbit, UK Biobank
accelerometer data).

### 3.2 How does this link to other papers and this work?

| Paper | Link to MyHeartCounts |
|---|---|
| Klasnja 2019 (HeartSteps V1) | The 37 HeartSteps V1 participants had similar demographics to MyHeartCounts. The 30-min step counts in HeartSteps are a fraction of the daily totals here. |
| PR #85 (HeartSteps V2 reproduction) | Uses NHANES data; mean ~6,000-7,000 steps/day. The MyHeartCounts mean is a sanity check on the NHANES-derived numbers. |
| PR #87 (simulator validation) | Recommends NHANES behavioural fingerprint matching. MyHeartCounts provides an *independent* reference. If the project's synthetic data matches MyHeartCounts (not just NHANES), the validation is stronger. |
| PR #86 (statistical analysis plan) | The primary outcome is cumulative steps over 42 days. MyHeartCounts provides the population reference for the cumulative-step distribution. |
| PR #89 (framework comparison) | MyHeartCounts is the largest *observational* mHealth dataset; StepCountJITAI is the largest *simulated* JITAI dataset. The project spans both. |

### 3.3 Is there still room for improvement?

Three improvements the project could make:

1. **Multi-source validation.** Currently PR #87 (simulator
   validation) recommends NHANES as the reference. Adding
   MyHeartCounts as a *secondary* reference (a different
   population, a different measurement device) would
   strengthen the validation. The two datasets give
   *independent* bounds on the synthetic distribution.

2. **Subgroup validation.** MyHeartCounts reports subgroup
   means by age, sex, BMI. The project's synthetic data
   should produce similar subgroup statistics. This is a
   1-day validation experiment.

3. **Demographic-specific calibration.** HeartSteps V1 had
   37 participants; the project could match the *demographic
   distribution* of MyHeartCounts (38% female, mean age 40,
   mean BMI 27) when calibrating the V1 simulator. This is
   a configuration change, not a code change.

### 3.4 Take action

Five concrete actions:

1. **Add MyHeartCounts as a secondary validation reference
   in PR #87 (simulator validation).** Currently only NHANES
   is referenced. MyHeartCounts provides an independent
   smartphone-based reference.

2. **In the project's synthetic data generator, set default
   `mean_daily_steps = 6000` and `σ = 2500`.** This matches
   MyHeartCounts. The current default is unspecified.

3. **For the project's framework, add a `validate.py` script
   that compares the synthetic distribution to MyHeartCounts
   summary stats.** This is a 1-day implementation task,
   not a research artefact. The recreation PR documents
   the validation target; the implementation lives in M-08.

4. **For the paper, add a "Population reference" subsection
   to the Methods.** State that the synthetic data is
   calibrated to match MyHeartCounts (and NHANES), with
   the matching criteria explicit. This is a 1-paragraph
   addition with high external validity value.

5. **Consider matching the HeartSteps V1 demographic
   distribution to MyHeartCounts.** 38% female, mean age 40,
   mean BMI 27. This is a config change, not a code change.
   The project can document this as "we calibrated to a
   US smartphone-equipped adult population."

---

## 4. What this recreation validates (be honest)

- Validates that the project's target population for synthetic
  data is consistent with a real observational study.
- Does NOT validate that the project's synthetic data *does*
  match — that requires a separate validation experiment.
- Does NOT validate the project's user simulator behaviour
  (only the marginal step-count distribution).

---

## 5. Numbers the project should track

| Number | MyHeartCounts | Project target (recommended) |
|---|---|---|
| Mean daily steps | ~6,000 | 6,000 |
| SD daily steps | ~2,500 (estimated) | 2,500 |
| Mean age | 38-42 | 40 |
| Female % | 37-42 | 40 |
| Mean BMI | 26-27 | 27 |
| 7-day retention | ~74% | 74% (synthetic users don't drop out by default) |

---

## 6. Open questions

- [ ] Is there a more recent (2020-2026) observational mHealth
      study with more detailed step-distribution data?
      HeartCounts, Fitbit, and All of Us are all candidates.
- [ ] Does the MyHeartCounts data show an effect of the iPhone
      (Stanford) phase vs. the national phase? Different
      demographics may produce different step distributions.
- [ ] What is the *between-individual* variance vs. the
      *between-day* variance in MyHeartCounts? This affects
      the project's user simulator calibration.

---

## 7. Citation block

```bibtex
@article{mcconnell2017myheart,
  author  = {McConnell, M. V. and others},
  title   = {My{H}eart{C}ounts: A mobile health study of cardiovascular health},
  journal = {The Lancet Digital Health},
  year    = {2017},
  doi     = {10.1016/S2589-7500(17)30012-4}
}
```

---

*End of MyHeartCounts recreation report v0.1*
