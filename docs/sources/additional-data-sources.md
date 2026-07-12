# Additional Dataset Survey: HeartSteps and Other Sources

**Date:** 2026-06-09
**Author:** William Dennis
**Context:** Supplementary to `sources/data-sources.md` — JITAI trial datasets, accessible benchmarks, and open wearable health data

---

## Purpose

`sources/data-sources.md` covers large-scale population datasets (All of Us, UK Biobank). Those are excellent for *population distributions* but (a) require 4-8 week access applications and (b) contain only passive sensing — no intervention delivery or user response data.

This document surveys datasets that **do include intervention delivery**, enabling direct calibration of the Subphase 1C user simulation. It also covers open-access wearable datasets for pipeline testing and synthetic data parameterisation.

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

**Protocol:** Spruijt-Metz D, Marlin BM, Pavel M, Rivera DE, Hekler E, De La Torre S, El Mistiri M, et al. (2022). "Advancing Behavioral Intervention and Theory Development for Mobile Health: The HeartSteps II Protocol." *Int J Environ Res Public Health*. 19(4):2267. doi: 10.3390/ijerph19042267. PMID: 35206455.

**Results papers:**
- De La Torre SA, El Mistiri M, Tung K, Hekler E, Klasnja P, Pavel M, Rivera DE, Spruijt-Metz D, Marlin B. (2024). "Modeling engagement with a digital behavior change intervention (HeartSteps II)." *J Biomed Inform*. 158:104721. doi: 10.1016/j.jbi.2024.104721. PMID: 39265816.
- De La Torre SA, El Mistiri M, Tung K, Hekler E, Klasnja P, Pavel M, Rivera DE, Spruijt-Metz D, Marlin B. (2025). "A dynamic Bayesian network approach to modeling engagement and walking behavior: insights from a yearlong micro-randomized trial (HeartSteps II)." *Health Psychol Behav Med*. 13(1):2552479. doi: 10.1080/21642850.2025.2552479. PMID: 40979180.
- Mistiri ME, De La Torre S, Marlin BM, Pavel M, Klasnja P, Spruijt-Metz D, Rivera DE. (2025). "Dynamic Modeling and System Identification of User Engagement in mHealth Interventions." *Control Eng Pract*. 164:106460. doi: 10.1016/j.conengprac.2025.106460. PMID: 40727918.

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

## Dataset 6: LiveWell

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

## Dataset 7: LifeSnaps

**Paper:** Hermans et al. (2023), *Scientific Data* (Nature)
**Lead:** Rik Hermans (TU Eindhoven) /生活方式分析团队

### Overview

| Property | Value |
|----------|-------|
| Type | Multi-modal Fitbit + smartphone sensing |
| Participants | 71 |
| Duration | 4 weeks (4 phases × 1 week each) |
| Data | Fitbit Charge 2 (HR, steps, sleep, calories), smartphone sensors (accelerometer, gyroscope, audio features), EMA self-reports |
| Labels | Mood, stress, energy, social context, sleep quality, physical activity |

### Why It Matters

EMA-coupled Fitbit data is rare and useful for fitting the *subjective response* component of the response model. Phased design (baseline → low activity → high activity → recovery) enables causal estimation of activity-mood coupling without HeartSteps access.

### Access

Open access via the 4TU ResearchData repository. DOI in Hermans et al. (2023).

---

## Dataset 8: NHANES Steps (Open-Access)

**Source:** [PhysioNet](https://physionet.org/content/minute-level-step-count-nhanes/)
**Year:** 2025

### Overview

| Property | Value |
|----------|-------|
| Type | Population-level accelerometry |
| Participants | 14,693 |
| Duration | 7 days per participant |
| Sensor | ActiGraph GT3X+ (80 Hz triaxial accelerometer) |
| Data | Step counts (5 algorithms), MIMS, activity counts, wear flags |
| Access | **Public domain (CC0)** — no application needed |

### Why It Matters

The largest publicly available step-count dataset. 14.7K participants with minute-level step counts from a validated accelerometer. Essential for parameterising the synthetic data generator's step distributions — age groups, sex stratification, activity levels. Freely downloadable.

### Access

- **Open access (CC0):** [PhysioNet](https://physionet.org/content/minute-level-step-count-nhanes/)
- **Format:** CSV
- **Size:** Large (14.7K participants × 7 days × minute-level)

---

## Dataset 9: TILES-2018 (Tracking IndividuaL performancE with Sensors)

> **Correction note (2026-06-17):** A previous version of this section cited the dataset as "TILES" hosted on OSF (`osf.io/jm6du/`) and led by Laura Stroud (Northwestern), claiming it contained an "intervention arm with delivery logs." All three claims are incorrect. The OSF ID does not resolve to a TILES dataset (OSF API returns 404 on `/v2/nodes/jm6du/` and `/v2/registrations/jm6du/`). The canonical source is Mundnich et al. 2020, *Scientific Data* 7:354 (DOI `10.1038/s41597-020-00655-3`). The dataset is observational — it has no `a_t` variable, so it cannot be used to learn `P(s_{t+1} | s_t, a_t)`. It is retained here for its passive-sensing value (Fitbit steps/HR/sleep), not for transition learning. See `docs/plans/learned-transitions.md` for the dataset selection that resulted from this correction.

**Source:** [tiles-data.isi.edu](https://tiles-data.isi.edu/)
**Lead:** Karel Mundnich, Shrikanth Narayanan (USC / USC ISI)
**Paper:** Mundnich et al. (2020), *Scientific Data* 7:354. DOI [10.1038/s41597-020-00655-3](https://doi.org/10.1038/s41597-020-00655-3)

### Overview

| Property | Value |
|----------|-------|
| Type | Longitudinal passive sensing (**observational only** — no intervention delivery) |
| Participants | n = 212 hospital workers |
| Duration | 10 weeks |
| Sensors | Fitbit Charge 2 (steps, sleep, HR), OMsignal biometrics garment, Jelly smartphone (audio features + Bluetooth proximity), Owl-in-One Bluetooth data hubs, Minew environmental sensors (door motion, humidity, temperature, light) |
| Surveys | Baseline + daily EMAs (job performance, health, personality, psychological flexibility, psychological capital) |
| Access | **Open** (USC ISI hosting) |

### Why It Matters

Two-thousand-plus hours of Fitbit (steps + HR + sleep) from 212 participants in a real workplace. Useful for:

- Parameterising synthetic step / HR / sleep distributions (with NHANES for steps, MMASH for HR-activity-sleep correlations).
- Modelling non-stationary behaviour over 10 weeks at the population level.
- Validating the data pipeline (1A) on long-duration Fitbit data.

### Why It Does NOT Help With Transition Learning

The dataset is observational. **There is no `a_t` variable** (no intervention delivered by an agent, no randomized arms). It cannot produce paired `(state, action, next_state)` tuples. For transition learning we use 4TU #1 "Collaboratively Setting Daily Step Goals" instead — see `docs/plans/learned-transitions.md`.

### Access

- **Open access:** [tiles-data.isi.edu](https://tiles-data.isi.edu/) (two records: Main Data Record + Audio Data Record)
- **Format:** CSV

---

## Dataset 10: GLOBEM (Generalizable Longitudinal Behavioral Monitoring)

**Source:** [PhysioNet](https://physionet.org/content/globem/1.1/)
**Year:** 2023

### Overview

| Property | Value |
|----------|-------|
| Type | Multi-year passive sensing + EMA |
| Participants | 497 unique (705 person-years) |
| Duration | 4 years (2018-2021), 10 weeks/year |
| Sensors | Phone (location, usage, Bluetooth, calls), Fitbit (steps, sleep) |
| Surveys | PHQ-4, PSS-4, PANAS, BDI-II, UCLA loneliness |
| Access | Credentialed (PhysioNet, DUA required) |

### Why It Matters

Multi-year passive sensing with Fitbit data. Covers behavioural patterns across academic terms — useful for understanding non-stationarity in activity data. The EMA surveys provide mental health context that could inform user simulation archetypes.

### Access

- **Credentialed access:** [PhysioNet](https://physionet.org/content/globem/1.1/)
- **Requires:** Data use agreement

---

## Dataset 11: PAMAP2 Physical Activity Monitoring

**Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
**Also on:** [HuggingFace](https://huggingface.co/datasets/rnicrosoft/PAMAP2)

### Overview

| Property | Value |
|----------|-------|
| Type | Multi-sensor activity recognition |
| Participants | 9 subjects |
| Duration | ~1.5 hours per subject |
| Sensors | 3 IMUs (hand, chest, ankle) + HR monitor (100 Hz) |
| Activities | 12 types (walking, running, cycling, housework, etc.) |
| Samples | 3.85M+ labeled data points |
| Access | **Open** |

### Why It Matters

The most widely-used wearable activity recognition benchmark. Multiple body-worn sensors with fine-grained activity labels. Useful for testing the data pipeline (1A) with multi-sensor data and validating activity classification in synthetic data.

### Access

- **Open access:** [UCI](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) or [HuggingFace](https://huggingface.co/datasets/rnicrosoft/PAMAP2)
- **Format:** CSV

---

## Dataset 12: MMASH (Multilevel Monitoring of Activity and Sleep)

**Source:** [PhysioNet](https://physionet.org/content/mmash/1.0.0/)
**Year:** 2020

### Overview

| Property | Value |
|----------|-------|
| Type | Multimodal monitoring (24h) |
| Participants | 22 healthy adults |
| Duration | 24 hours continuous |
| Sensors | Polar H7 (beat-to-beat HR), ActiGraph (tri-axial accel, steps) |
| Additional | Sleep quality, questionnaires, salivary cortisol/melatonin |
| Access | **Open (ODbL)** |

### Why It Matters

Rich multimodal data: heart rate, acceleration, sleep, and biomarkers — all in one dataset. Freely downloadable. Good for validating that synthetic data produces realistic HR-activity-sleep correlations.

### Access

- **Open access (ODbL):** [PhysioNet](https://physionet.org/content/mmash/1.0.0/)

---

## Dataset 13: Health Gym (Synthetic RL Environments)

**Source:** [GitHub](https://github.com/nikooo777/health-gym)
**Paper:** Gateno et al. (2023), *Nature Scientific Data*

### Overview

| Property | Value |
|----------|-------|
| Type | Synthetic RL environments from MIMIC-III |
| Environments | HIV treatment, Sepsis, Acute Hypotension |
| Format | CSV (state-action-reward trajectories) |
| Access | **Open (GitHub, MIT)** |

### Why It Matters

Pre-built synthetic RL environments designed specifically for offline RL algorithm development. While clinical (not PA-focused), the trajectory format and reward structure are directly transferable to our framework design. Useful for testing the agent library (1D) before PA-specific data is available.

### Access

- **Open access (MIT):** [GitHub](https://github.com/nikooo777/health-gym)

---

## Gap Analysis: All Dataset Categories

| Need | Population (All of Us / UKB) | Trial (HeartSteps V1/V2) | Open Benchmark (WISDM, ExtraSensory, PAMAP2) | Open Wearable (NHANES, MMASH, TILES) | Cardio Trial (MyHeartCounts) | Wearable Camera / EMA (LiveWell, LifeSnaps) |
|---|---|---|---|---|---|---|
| Step count distributions | ✅ Large N | ⚠️ Small N | ✅ Activity-specific | ✅ NHANES 14.7K | ✅ Apple Watch | ✅ LiveWell (cam), LifeSnaps (Fitbit) |
| Heart rate | ✅ All of Us | ❌ | ❌ | ✅ MMASH (beat-to-beat) | ✅ Apple Watch (continuous) | ✅ LiveWell (Fitbit+cam) |
| Sleep | ✅ | ⚠️ V2 only | ❌ | ✅ MMASH, TILES | ✅ Apple Watch | ✅ LifeSnaps (Fitbit) |
| Intervention response | ❌ | ✅ | ❌ | ❌ (TILES is observational) | ⚠️ Custom (non-PA prompts) | ⚠️ LiveWell (eating cues) |
| Engagement over time | ❌ | ✅ | ❌ | ✅ TILES (10 weeks) | ✅ 7-day studies | ✅ LifeSnaps (4 weeks × 4 phases) |
| RL action selection data | ❌ | ✅ V2 | ❌ | ❌ | ❌ | ❌ |
| Immediate availability | ❌ (4-8 week apps) | ⚠️ (data sharing) | ✅ (download now) | ✅ NHANES, MMASH, TILES (now) | ⚠️ (Stanford DUA) | ✅ LiveWell (Alshurafa lab), LifeSnaps (4TU open) |
| Longitudinal (months+) | ✅ (years) | ✅ (42-90 days) | ❌ (days) | ✅ GLOBEM (4 yr), TILES (10 wk — borderline) | ❌ (7 days) | ⚠️ LiveWell (4 mo), LifeSnaps (4 wk) |
| Sensor variety | ✅ Fitbit | ✅ Fitbit | ✅ Phone + IMU | ✅ Mixed (Fitbit, ActiGraph, Polar) | ✅ Apple Watch | ✅ LiveWell (cam+Fitbit), LifeSnaps (Fitbit+phone) |

---

## Recommendation

### For Phase 1 (Now)

| Priority | Dataset | Use | Access |
|---|---|---|---|
| 1 | NHANES Steps | Parameterise synthetic step distributions (14.7K participants, CC0) | Download now |
| 2 | ExtraSensory / WISDM / PAMAP2 | Test data pipeline, validate schemas | Download now |
| 3 | MMASH | Validate HR-activity-sleep correlations in synthetic data | Download now |
| 4 | TILES-2018 | Passive Fitbit (steps/HR/sleep) for synthetic-distribution parameterisation; observational only — see `docs/plans/learned-transitions.md` | Download now |
| 4b | 4TU #1 "Collaboratively Setting Daily Step Goals" | Open intervention-response data (step-goal proposals) for 1C calibration | Download now |
| 5 | Health Gym | Test agent library (1D) with synthetic RL trajectories | Download now |
| 6 | HeartSteps V1 | Best restricted intervention response data — start access request | Weeks-months |
| 7 | HeartSteps V2 | RL-in-the-loop data for agent benchmark — start access request | Weeks-months |
| 8 | All of Us / UK Biobank | Population distributions for synthetic data — applications in parallel | 4-8 weeks |

### Impact on Plan

1. **Subphase 1A (Data Layer):** NHANES (14.7K participants, CC0) provides the primary source for parameterising synthetic step distributions. PAMAP2 and WISDM validate the pipeline on real sensor data on Day 1. MMASH validates HR-activity-sleep correlations.

2. **Subphase 1C (User Simulation):** 4TU #1 "Collaboratively Setting Daily Step Goals" (235 participants, 5-level RL-randomised goal proposal, prior-day step response) is the open-access intervention dataset for *action → response* mappings without needing HeartSteps access. TILES-2018 is observational and *cannot* supply intervention response data, despite an earlier version of this document claiming otherwise (see correction note in Dataset 9 above). HeartSteps V1/V2 remains the gold standard for burden/engagement calibration.

3. **Subphase 1D (Agent Library):** Health Gym provides ready-made synthetic RL environments for testing agent implementations before PA-specific data is available. HeartSteps V2's Thompson Sampling is the real benchmark.

4. **Synthetic data strategy:** Fit synthetic generators to NHANES step distributions (population baseline) + HeartSteps / 4TU #1 response distributions (treatment effects). MMASH grounds the HR-activity-sleep correlations. TILES-2018 Fitbit (steps/HR/sleep) provides 10-week real-world distributions of passive wearable signals.

---

## Files Reference

| File | Purpose |
|------|---------|
| `sources/data-sources.md` | Large-scale population dataset feasibility study |
| This document | JITAI trial datasets and accessible sources |
| `initial_design.tex` | Formal MDP definition with state variables |
| `code/codebase-plan.md` | Architecture and phased delivery plan |
