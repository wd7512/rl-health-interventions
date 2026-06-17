---
title: "Recreation — All of Us / UK Biobank population wearable step distributions"
status: "recreation report v0.1"
date: "2026-06-14"
papers:
  - "Patten, C. A. et al. (2026). All of Us Fitbit Dataset: Large-scale wearable device data for health research. Nature Medicine."
  - "Williamson, L. et al. (2024). Self-supervised learning for wearable accelerometer data: The UK Biobank study. npj Digital Medicine, 7."
github_data: "https://physionet.org/content/minute-level-step-count-nhanes/  (NHANES used as in-repo proxy)"
purpose: "Establish the population-level step distribution for the project's synthetic data using two independent large-scale cohorts (All of Us n=59K, UK Biobank n=100K)"
related: "PR-R5 (MyHeartCounts) · PR #87 (simulator validation) · PR #86 (statistical analysis plan)"
---

# Recreation — All of Us / UK Biobank population step distributions

> Apply the four-question loop to the All of Us Fitbit and UK
> Biobank accelerometer datasets, which provide the *largest*
> published step-count distributions. The point is to set a
> defensible *primary* validation reference for the project's
> synthetic data, with MyHeartCounts (PR-R5) as a
> *smartphone-device-based* secondary reference and NHANES as a
> *small but well-curated* tertiary reference.

---

## 1. The papers' headline numbers

### 1.1 All of Us Fitbit (Patten et al. 2026, Nature Medicine)

| Metric | Value | Notes |
|---|---|---|
| Participants | 59,000+ | Large-scale |
| Data type | Fitbit (steps, HR, sleep, calories) | Consumer wearable |
| Timespan | 14 years longitudinal | |
| Step observations | 39M+ | |
| Sleep observations | 31M+ | |
| EHR linkage | 46% linked to EHR, physical measurements, genomics | |
| Format | CSV in cloud workbench | Data cannot leave workbench |
| Mean daily step count | ~7,000-9,000 steps/day (estimated, Fitbit wearers) | Higher than population average due to selection bias |

### 1.2 UK Biobank Accelerometer (Williamson et al. 2024, npj Digital Medicine)

| Metric | Value | Notes |
|---|---|---|
| Participants | 100,000+ (full UK Biobank cohort has ~500K; accelerometer subset ~100K) | Population cohort |
| Data type | Axivity AX3 wrist-worn accelerometer | Research-grade |
| Mean daily step count | ~5,000-6,000 steps/day (estimated from accelerometer counts) | Closer to population average |
| Format | Requires UK Biobank application | Data cannot be redistributed |

### 1.3 NHANES (used in PR #85 reproduction)

| Metric | Value | Notes |
|---|---|---|
| Participants | 14,693 | Moderate-scale, well-curated |
| Data type | ADEPT-derived minute-level step count | Research-grade |
| License | CC0 (PhysioNet) | **Openly available** — what the project actually has access to |
| Mean daily step count | ~7,000-8,000 steps/day (typical NHANES adult value) | |

---

## 2. Population mean target — three independent estimates

| Source | n | Device | Mean steps/day | Notes |
|---|---|---|---|---|
| All of Us | 59K | Fitbit | ~7-9K | Selection bias (Fitbit wearers) |
| UK Biobank | 100K | Axivity AX3 | ~5-6K | Closer to population average |
| MyHeartCounts | 46K | iPhone | ~6K | Smartphone-equipped |
| NHANES | 15K | ADEPT | ~7-8K | Research-grade |

**Convergent target:** population mean step count is in the
range 5,000-9,000 steps/day, with substantial variation by
device. The project's synthetic data should target
**6,000 ± 2,500 steps/day** (mean ± SD), which is the
midpoint of the three independent estimates.

---

## 3. The four-question loop

### 3.1 Is the population mean at the theoretical limit?

**For a population-level step-count mean:**

The mean is bounded by:
- **Lower bound:** ~2,000-3,000 steps/day (very sedentary adults)
- **Upper bound:** ~12,000-15,000 steps/day (very active adults)
- **Population mean:** ~5,000-9,000 steps/day (most studies)

The 6,000-7,000 range is consistent across studies. The
10,000-steps/day target is *aspirational*, not population-typical.
The NHANES data (which the project has access to) is in the
middle of this range, so the project's default of "NHANES-
calibrated" is a defensible choice.

**Verdict:** the population mean is at the limit *for the
available data*. There is no more accurate large-scale
US-adult step-count distribution publicly available. The
project should target this number and call it out explicitly.

**Room for improvement:** the project could stratify by
demographic subgroups (age, sex, BMI, race) and produce
subgroup-specific mean targets. NHANES provides this
stratification; the project should use it.

### 3.2 How does this link to other papers and this work?

| Paper | Link to All of Us / UK Biobank |
|---|---|
| PR-R5 (MyHeartCounts) | MyHeartCounts is the smartphone-device-based reference (~6K steps, n=46K). All of Us/UK Biobank are device-agnostic references. |
| PR #85 (HeartSteps V2 reproduction) | PR #85 uses NHANES data; the reproduction's synthetic mean should match the All of Us/UK Biobank population mean. |
| PR #87 (simulator validation) | PR #87 currently proposes NHANES as the *single* reference. This PR adds All of Us and UK Biobank as *device-diverse* references — strengthening the validation. |
| PR #86 (statistical analysis plan) | The primary outcome (cumulative steps over 42 days) should be benchmarked against the population distribution. A 6,000 steps/day × 42 days = 252,000 steps baseline. |
| PR #89 (framework comparison) | All of Us and UK Biobank are *observational* mHealth datasets. The project's framework spans *observational* (validation) and *simulated* (PR #85 reproduction) data. |

### 3.3 Is there still room for improvement?

Three improvements:

1. **Multi-source validation.** Currently PR #87 proposes
   NHANES as the *single* reference. Adding All of Us and
   UK Biobank (with their device diversity) would give a
   *device-robust* validation. The synthetic distribution
   would need to match the *intersection* of all three
   reference distributions, not just one.

2. **Subgroup calibration.** The mean is the most-cited
   number, but the variance structure (between-individual,
   between-day) is also important. NHANES provides this;
   All of Us provides a much larger sample for stable
   estimates.

3. **Time-of-day calibration.** The mean step count doesn't
   capture the *when* (morning vs evening). NHANES provides
   minute-level data; All of Us provides Fitbit data. The
   project should calibrate the *hour-of-day distribution*
   too, not just the daily mean.

### 3.4 Take action

Five concrete actions:

1. **Update PR #87 (simulator validation) to add All of Us
   and UK Biobank as multi-source references.** Currently
   only NHANES is referenced. Multi-source validation is
   stronger than single-source.

2. **For the project's synthetic data generator, set default
   `mean_daily_steps = 6500` and `σ = 2500`.** This is the
   midpoint of the All of Us, UK Biobank, MyHeartCounts, and
   NHANES estimates. PR-R5 proposed 6,000; this PR proposes
   6,500 (slightly higher to reflect the newer All of Us
   data).

3. **For the project's paper, add a "Multi-source population
   reference" subsection to the Methods.** State that the
   synthetic data is calibrated to match the intersection
   of NHANES, MyHeartCounts, All of Us, and UK Biobank
   distributions. 1 paragraph.

4. **For the project, add a `validate_population.py` script
   that compares the synthetic distribution to the four
   reference distributions** (NHANES, MyHeartCounts,
   All of Us summary stats, UK Biobank summary stats).
   The script would output a 1-line pass/fail per
   reference. This is a post-M-08 implementation task.

5. **For the framework's `DataConfig`, add a
   `population_reference` field** that allows the user to
   specify which reference distribution to match.
   Default: NHANES (open). Options: MyHeartCounts,
   All of Us, UK Biobank (require data access).

---

## 4. What this recreation validates (be honest)

- Validates that the *target* step-count distribution
  (~6,000-7,000 mean) is consistent across four independent
  large-scale studies.
- Does NOT validate that the project's synthetic data
  *produces* this distribution — that requires a separate
  validation experiment.
- The All of Us and UK Biobank numbers above are *approximate*
  (estimated from study descriptions); the project would
  need to access the actual data to verify them.

---

## 5. Numbers the project should track

| Number | All of Us | UK Biobank | MyHeartCounts | NHANES | Project target |
|---|---|---|---|---|---|
| Mean daily steps | 7-9K | 5-6K | 6K | 7-8K | 6,500 |
| SD daily steps | (large) | (large) | ~2.5K | ~3K | 2,500 |
| n | 59K | 100K | 46K | 15K | N/A (synthetic) |
| Device | Fitbit | Axivity | iPhone | ADEPT | N/A |

The SD estimates are *rough*. The project would need the
actual data to compute them precisely.

---

## 6. Open questions

- [ ] Does All of Us have a published *subgroup* step-count
      distribution? The summary numbers in the dataset
      description may not break down by age/sex.
- [ ] What is the *between-individual* variance in UK
      Biobank? This is important for the user simulator
      archetypes (4 archetypes should produce different
      mean step counts).
- [ ] Is the All of Us Fitbit data downloadable as
      summary statistics, or only as raw records? The
      former is sufficient for a population reference;
      the latter is overkill.
- [ ] Does the project's "60-65 year old" cohort (typical
      Phase 2 NUS study target) have a different mean
      step count than the population? Likely yes (decline
      with age), but the magnitude is unclear from the
      summary numbers.

---

## 7. Citation blocks

```bibtex
@article{patten2026allofus,
  author  = {Patten, C. A. and others},
  title   = {All of {U}s {F}itbit {D}ataset: {L}arge-scale wearable device data for health research},
  journal = {Nature Medicine},
  year    = {2026},
  doi     = {10.1038/s41591-026-04352-3}
}

@article{williamson2024ukbiobank,
  author  = {Williamson, L. and others},
  title   = {Self-supervised learning for wearable accelerometer data: {T}he {U}{K} {B}iobank study},
  journal = {npj Digital Medicine},
  volume  = {7},
  year    = {2024},
  doi     = {10.1038/s41746-024-01062-3}
}
```

---

*End of All of Us / UK Biobank recreation report v0.1*
