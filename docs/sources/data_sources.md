# Public Dataset Feasibility Study

**Date:** 2026-06-09
**Author:** William Dennis
**Context:** Bristol x NUS RL Health Internship — Phase 1 data layer planning

---

## Purpose

Evaluate two public wearable health datasets as potential sources for training
the rule-based user simulator (Subphase 1C). This document captures schema
details, access requirements, usage by other researchers, and a gap analysis
against our MDP specification.

**Conclusion:** Both datasets require institutional applications with 4-8 week
lead times. Neither provides downloadable samples. The framework should target
synthetic data generation for Phase 1, with these schemas as the ingestion
target when real data becomes available.

---

## Dataset 1: All of Us Fitbit Dataset

**Paper:** [Nature Medicine 2026](https://www.nature.com/articles/s41591-026-04352-3)

### Overview

| Property | Value |
|----------|-------|
| Participants | 59,000+ |
| Data type | Fitbit (steps, heart rate, sleep, calories) |
| Timespan | 14 years (longitudinal) |
| Step observations | 39M+ |
| Sleep observations | 31M+ |
| EHR linkage | 46% linked to EHR, physical measurements, genomics |
| Format | CSV (in cloud workbench) |

### Access

1. Register at [researchallofus.org](https://researchallofus.org/)
2. Complete CITI training + Data or Specimens Only certification
3. Submit research application (requires institutional affiliation)
4. Access via All of Us Researcher Workbench (cloud-based, no local download)

**Key constraint:** Data cannot leave the workbench. All analysis runs in their
cloud environment. We cannot download CSVs to test locally.

### Schema (from data dictionary)

| Field | Column Name | Type | Unit |
|-------|-------------|------|------|
| Participant ID | `person_id` | string | — |
| Date | `date` | datetime | — |
| Steps | `steps` | integer | count/day |
| Resting Heart Rate | `resting_heart_rate` | float | bpm |
| Heart Rate Minutes | `heart_rate_minutes` | float | minutes |
| Sleep Minutes | `sleep_minutes` | float | minutes |
| Sleep Efficiency | `sleep_efficiency` | float | ratio (0-1) |
| Calories | `calories` | float | kcal |
| Distance | `distance_meters` | float | meters |
| Floors Climbed | `floors_climbed` | integer | count |

### How Others Have Used It

- **Physical activity patterns:** Step count distributions across demographics
- **Sleep-health associations:** Linking sleep duration/quality to health outcomes
- **Longitudinal trends:** Tracking activity changes over years
- **Demographic disparities:** Activity levels across age, gender, ethnicity
- **Clinical correlations:** Linking wearable data to EHR diagnoses

### Relevance to Our Framework

| MDP Variable | Available? | Notes |
|---|---|---|
| `steps_t` | ✅ Direct | Daily step counts |
| `hr_t` | ✅ Direct | Resting heart rate |
| `sleep_hours_t` | ✅ Direct | Sleep minutes (convert to hours) |
| `sedentary_min_t` | ⚠️ Indirect | Could derive from low-activity periods |
| `time_of_day_t` | ⚠️ Indirect | Derivable from intraday/minute-level tables (heart_rate_minute_level, steps_intraday) |
| `age`, `gender` | ✅ Via demographics | Separate table, joinable |
| `body_measure_k` | ❌ | Not in Fitbit data |

---

## Dataset 2: UK Biobank Accelerometer Dataset

**Paper:** [npj Digital Medicine 2024](https://www.nature.com/articles/s41746-024-01062-3)
**Code:** [OxWearables/ssl-wearables](https://github.com/OxWearables/ssl-wearables)

### Overview

| Property | Value |
|----------|-------|
| Participants | 100,000+ |
| Person-days | 700,000+ |
| Recording duration | 7 days per participant |
| Sampling rate | 100 Hz (tri-axial accelerometer) |
| Format | CSV (via UK Biobank RAP) |

### Access

1. Register at [ukbiobank.ac.uk](https://www.ukbiobank.ac.uk/register-apply/)
2. Submit Research Application (requires institutional affiliation)
3. Pay application fee (~£500-9000 depending on scope)
4. Data delivered via UK Biobank Research Analysis Platform

**Key constraint:** Application takes 4-8 weeks. Data access is via their
cloud platform. Local download possible after approval.

### Schema (from data dictionary)

The UK Biobank data comes at two levels — daily summaries and raw epoch-level
accelerometer data. These are structurally different and should be treated
separately for ingestion.

#### Daily Summary Data

| Field | Column Name | Type | Unit |
|-------|-------------|------|------|
| Participant ID | `eid` | string | — |
| Date | `date` | datetime | — |
| Steps | `steps` | integer | count/day |
| Wear Time | `wear_time` | float | minutes |

#### Raw/Epoch Accelerometer Time-Series (100 Hz)

| Field | Column Name | Type | Unit |
|-------|-------------|------|------|
| Participant ID | `eid` | string | — |
| Timestamp | `timestamp` | datetime | — |
| Accelerometer X | `x` | float | g |
| Accelerometer Y | `y` | float | g |
| Accelerometer Z | `z` | float | g |
| Angle Z | `anglez` | float | degrees |
| ENMO | `enmo` | float | mg |
| Pitch | `pitch` | float | degrees |
| Roll | `roll` | float | degrees |
| Light | `light` | float | lux |
| Temperature | `temperature` | float | celsius |

**ENMO** (Euclidean Norm Minus One) is the standard metric for movement
intensity. Values near 0 = sedentary, higher = more active.

### How Others Have Used It

- **Sleep/wake classification:** OxWearables SSL models trained on this data
- **Physical activity quantification:** ENMO as activity intensity measure
- **Sedentary behaviour research:** Identifying sedentary bouts from accelerometer
- **Circadian rhythm analysis:** Using ENMO patterns to detect sleep/wake cycles
- **Mortality/morbidity associations:** Linking activity patterns to health outcomes

### Relevance to Our Framework

| MDP Variable | Available? | Notes |
|---|---|---|
| `steps_t` | ✅ Direct | Derived from accelerometer |
| `hr_t` | ❌ | No heart rate in this dataset |
| `sleep_hours_t` | ⚠️ Derived | OxWearables models can classify sleep |
| `sedentary_min_t` | ✅ Direct | ENMO < threshold = sedentary |
| `time_of_day_t` | ✅ Direct | Hourly timestamps from 100Hz data |
| `age`, `gender` | ✅ Via demographics | Separate UK Biobank table |
| `body_measure_k` | ❌ | Not in accelerometer data |

---

## Gap Analysis: Datasets vs MDP Specification

The MDP spec (design.tex) defines these state variables:

| Variable | All of Us | UK Biobank | Both Combined | Source |
|---|---|---|---|---|
| `steps_t` | ✅ | ✅ | ✅ | Direct from either |
| `hr_t` | ✅ | ❌ | ✅ | All of Us only |
| `sleep_hours_t` | ✅ | ⚠️ | ✅ | All of Us direct, UK Biobank via model |
| `sedentary_min_t` | ⚠️ | ✅ | ✅ | UK Biobank ENMO |
| `time_of_day_t` | ⚠️ | ✅ | ✅ | All of Us intraday / UK Biobank hourly |
| `day_of_week_t` | ✅ | ✅ | ✅ | Derivable from date |
| `goal_progress_t` | ❌ | ❌ | ❌ | Synthetic/simulated |
| `burden_t` | ❌ | ❌ | ❌ | State variable (not from data) |
| `body_measure_k` | ❌ | ❌ | ❌ | Needs EHR (All of Us has 46% linked) |
| `age` | ✅ | ✅ | ✅ | Demographics tables |
| `gender` | ✅ | ✅ | ✅ | Demographics tables |
| `baseline_activity` | ❌ | ❌ | ❌ | Derived from step distributions |
| `previous_action` | ❌ | ❌ | ❌ | State variable (not from data) |
| `previous_response` | ❌ | ❌ | ❌ | State variable (not from data) |

**Key finding:** Between the two datasets, we can cover most wearable-derived
MDP variables. The gaps are:
- **Goal progress, burden, previous action/response** — these are state
  variables that evolve during simulation, not from raw data
- **Body measures** — would need EHR linkage (All of Us has 46% linked)
- **Baseline activity** — derived from step distributions, not a raw field

---

## Recommendation

### Phase 1 (Now): Synthetic Data Generation

Since neither dataset is immediately accessible, Phase 1 should focus on:

1. **Synthetic data generator** that produces realistic wearable data matching
   the schemas above (step distributions, heart rate ranges, sleep patterns)
2. **Config-driven ingestion** ready to swap in real data when access is granted
3. **Distribution fitting** — use published summary statistics from both papers
   to parameterise synthetic generators

### Phase 2 (After Access): Real Data Integration

Once institutional access is obtained:

1. **All of Us** — apply for researcher workbench access (cloud-based)
2. **UK Biobank** — submit research application (4-8 weeks)
3. Swap synthetic data for real data by changing the config YAML
4. Validate simulator behaviour against real distributions

### Dataset Selection

| Use Case | Recommended Dataset |
|---|---|
| Heart rate + steps + sleep | All of Us |
| Sedentary behaviour + hourly patterns | UK Biobank |
| Both HR and sedentary | Need both (complementary) |
| Quick prototyping | Synthetic data (now) |

---

## Files Reference

| File | Purpose |
|------|---------|
| `design.tex` | Formal MDP definition with state variables |
| `code/codebase_plan.md` | Architecture and phased delivery plan |
| This document | Dataset feasibility and gap analysis |
