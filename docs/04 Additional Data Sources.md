# Additional Dataset Survey: HeartSteps and Other Sources

**Date:** 2026-06-09
**Author:** William Dennis
**Context:** Supplementary to `docs/03 Data Sources.md` — JITAI trial datasets and smaller accessible sources

---

## Purpose

`docs/03 Data Sources.md` covers large-scale population datasets (All of Us, UK Biobank). Those are excellent for *population distributions* but (a) require 4-8 week access applications and (b) contain only passive sensing — no intervention delivery or user response data.

This document surveys datasets that **do include intervention delivery**, enabling direct calibration of the Subphase 1C user simulation. Most are smaller and some are publicly available immediately.

---

## Dataset 1: HeartSteps V1

**Paper:** Klasnja et al. (2019), *Annals of Behavioral Medicine*
**Lead:** Predrag Klasnja / Susan Murphy (University of Michigan)

### Overview

| Property | Value |
|----------|-------|
| Type | Micro-randomized trial (MRT) |
| Participants | ~50 adults |
| Duration | 42 days per participant |
| Decision frequency | 5x/day (contextual) |
| Intervention type | Activity suggestions, anti-sedentary messages |
| Primary outcome | Daily step count (Fitbit) |

### What It Contains

- Step counts at 30-min granularity (Fitbit)
- Intervention delivery history (type, timing, context)
- User engagement / dismissal of suggestions
- Contextual features: time of day, location (GPS-derived), day of week, weather
- Self-report surveys (motivation, fatigue, stress)

### Relevance to Our Framework

| MDP Component | How HeartSteps V1 Helps |
|---|---|
| State space | Real context features validate our feature set |
| Action space | Suggestion types map to our action set |
| User response model | Observed responses — direct calibration for 1C |
| Fatigue / burden | Engagement decay over 42 days — real dynamics |
| Baseline activity | Pre-trial step distributions |

### Access

Part of the [HeartSteps research project](https://klasnja.org/heartsteps/). Data requests via the Klasnja lab at University of Michigan. Some de-identified data available on [Open Science Framework](https://osf.io/).

**Timeline:** Weeks to months (data sharing agreement).

---

## Dataset 2: HeartSteps V2

**Paper:** Klasnja et al. (2021/2022), *npj Digital Medicine* (expected)
**Lead:** Predrag Klasnja / Susan Murphy (University of Michigan)

### Overview

| Property | Value |
|----------|-------|
| Type | Micro-randomized trial (MRT) with RL component |
| Participants | ~97 adults |
| Duration | 90 days per participant |
| Decision frequency | 5x/day |
| Intervention type | Contextually-tailored activity suggestions |
| Primary outcome | Daily step count (Fitbit) |

### What Makes V2 Different from V1

- **Larger and longer:** 97 participants × 90 days vs 50 × 42
- **RL-based tailoring:** Contextual bandit (Thompson Sampling) — real RL-in-the-loop data
- **Richer context:** More location features, weather, calendar data
- **Multiple intervention types:** Walking suggestions, goal reminders, anti-sedentary nudges, motivational messages
- **Sleep tracking:** Added sleep data collection

### Relevance

| MDP Component | How HeartSteps V2 Helps |
|---|---|
| Thompson Sampling agent | Real benchmark for 1D |
| Reward signals | Step count deltas ground our `R_t` weights |
| Non-stationarity | 90-day engagement curves |
| Multi-action evaluation | Different suggestion effect sizes |
| Delayed effects | Lagged step-count responses |

### Access

Same as V1 — Klasnja lab, University of Michigan. V2 data is more structured.

---

## Dataset 3: ExtraSensory

**Paper:** Vaizman et al. (2017), *UbiComp*
**Link:** [extrasensory.ucsd.edu](http://extrasensory.ucsd.edu/)

### Overview

| Property | Value |
|----------|-------|
| Type | Passive sensing + self-report labels |
| Participants | 60 |
| Duration | ~7 days each |
| Sensors | Phone accelerometer, gyroscope, compass, audio, location |
| Labels | Activity type, mood, social context, phone usage |

### Why It Matters

Publicly downloadable **today** (no application needed). Contains labelled activity data that can seed the synthetic data generator's feature distributions. Not an intervention dataset, but useful for validating the data pipeline (1A) on real sensor data.

### Access

- **Open access:** [http://extrasensory.ucsd.edu/](http://extrasensory.ucsd.edu/)
- **Size:** ~6 GB of phone sensor data
- **Format:** CSV files per user

---

## Dataset 4: WISDM

**Paper:** Kwapisz et al. (2011), *ACM SIGKDD Explorations*
**Link:** [www.cis.fordham.edu/wisdm/](https://www.cis.fordham.edu/wisdm/)

### Overview

| Property | Value |
|----------|-------|
| Type | Labelled activity recognition |
| Participants | 36 |
| Activities | Walking, jogging, stairs, sitting, standing, lying |
| Sensor | Phone accelerometer (20 Hz) |
| Size | ~1 GB |

### Why It Matters

The most widely-used activity recognition benchmark. Useful for testing the Polars pipeline (1A) with known ground-truth labels, validating synthetic data distributions, and quick prototyping.

### Access

- **Open access:** [WISDM dataset](https://www.cis.fordham.edu/wisdm/dataset.php)

---

## Dataset 5: MyHeartCounts

**Paper:** McConnell et al. (2017), *The Lancet Digital Health*
**Lead:** Euan Ashley (Stanford)

### Overview

| Property | Value |
|----------|-------|
| Type | mHealth observational study |
| Participants | ~48,000 |
| Duration | Varies (self-selected) |
| Data | Apple Watch health data, surveys, 6-min walk test |
| Outcome | Cardiovascular fitness |

### Why It Matters

Large-scale Apple Watch data for population-level step distributions, heart rate variability patterns, and validating synthetic data ranges. Passive sensing only.

### Access

Data available via [Stanford Digital Health](https://myheartcounts.stanford.edu/) through a data use agreement.

---

## Dataset 6: LiveWell / LifeSnaps

**Paper:** Alshurafa et al. (2019), *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies*
**Lead:** Nabil Alshurafa (Northwestern)

### Overview

| Property | Value |
|----------|-------|
| Type | Behavioral monitoring with wearable cameras |
| Participants | 10 |
| Duration | 4 months |
| Data | Wearable accelerometer, heart rate, GPS, camera images |
| Labels | Eating, physical activity, social interactions |

### Why It Matters

Long-duration (4 months) with rich context. Useful for understanding behavioural pattern drift over time — important for the non-stationarity dimension of our MDP.

### Access

Contact the Alshurafa lab at Northwestern University.

---

## Gap Analysis: Trial Datasets vs Population Datasets

| Need | Population (All of Us / UKB) | Trial (HeartSteps V1/V2) | Accessible (WISDM, ExtraSensory) |
|---|---|---|---|
| Step count distributions | ✅ Large N | ⚠️ Small N | ✅ Activity-specific |
| Heart rate | ✅ All of Us | ❌ | ❌ |
| Sleep | ✅ | ⚠️ V2 only | ❌ |
| Intervention response | ❌ | ✅ | ❌ |
| Engagement over time | ❌ | ✅ | ❌ |
| RL action selection data | ❌ | ✅ V2 | ❌ |
| Immediate availability | ❌ (4-8 week apps) | ⚠️ (data sharing) | ✅ (download now) |
| Longitudinal (months+) | ✅ (years) | ✅ (42-90 days) | ❌ (days) |
| Sensor variety | ✅ Fitbit | ✅ Fitbit | ✅ Phone sensors |

---

## Recommendation

### For Phase 1 (Now)

| Priority | Dataset | Use |
|---|---|---|
| 1 | ExtraSensory / WISDM | Quick-start: test data pipeline, validate schemas, no access barriers |
| 2 | HeartSteps V1 | Best available *intervention response* data — start access request now |
| 3 | HeartSteps V2 | RL-in-the-loop data for agent benchmark — start access request now |
| 4 | All of Us / UK Biobank | Population distributions for synthetic data — access applications in parallel |

### Impact on Plan

1. **Subphase 1A (Data Layer):** Test the Polars ingestion pipeline against WISDM and ExtraSensory CSV files on Day 1. This catches schema mapping bugs before any data is at stake.

2. **Subphase 1C (User Simulation):** HeartSteps V1/V2 provides the only direct source of *action → response* mappings we have. Even small-N, these are more valuable for calibrating `ResponseModel` archetypes than population-level step distributions alone. The fatigue/burden decay curves from V1's 42-day engagement data are the best available signal for our burden mechanism.

3. **Subphase 1D (Agent Library):** HeartSteps V2's Thompson Sampling implementation and step-count outcomes provide a real-data benchmark for our agent. We can compare our agent's regret curves against V2's published results.

4. **Synthetic data strategy:** Fit synthetic generators to *both* population distributions (All of Us / UKB) *and* trial response distributions (HeartSteps). The former gives realistic baseline features; the latter gives realistic treatment effects.

---

## Files Reference

| File | Purpose |
|------|---------|
| `docs/03 Data Sources.md` | Large-scale population dataset feasibility study |
| This document | JITAI trial datasets and accessible sources |
| `docs/02 MDP Specification.tex` | Formal MDP definition with state variables |
| `docs/01 Codebase Plan.md` | Architecture and phased delivery plan |
