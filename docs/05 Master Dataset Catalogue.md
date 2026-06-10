# Master Dataset Catalogue

**Date:** 2026-06-10
**Context:** RL Health Interventions — Complete inventory of all known datasets

---

## Access Legend

| Rating | Meaning |
|--------|---------|
| **Open (download now)** | Publicly downloadable, no application needed |
| **Open (credential required)** | PhysioNet/4TU credential or data use agreement, but self-service |
| **Institutional (DURA)** | Requires institutional affiliation + formal application + 4-8 week review |
| **Restricted (contact lab)** | Must contact the research team directly for access |
| **Academic use** | Available but limited to academic/research use |

## MDP Variable Key

- S = steps_t (daily step count)
- H = hr_t (heart rate)
- L = sleep_hours_t (sleep duration)
- E = sedentary_min_t (sedentary minutes)
- T = time_of_day_t / day_of_week_t (temporal features)
- G = goal_progress_t (goal progress)
- B = burden_t (intervention burden)
- A = previous_action / response (intervention history)
- D = demographics (age, gender)
- K = body_measure_k (clinical measures)
- I = intervention delivery + response data

---

## Complete Dataset Inventory

| # | Dataset | N | Sensors | Duration | Access | Config PR | MDP Vars | Use Case |
|---|---------|---|---------|----------|--------|-----------|----------|----------|
| | **TIER 1: INTERVENTION + RL DATA** | | | | | | | |
| 1 | HeartSteps V1 | ~50 | Fitbit (steps, 30min) | 42 days | Restricted (contact lab) | — | S, T, I, A | Gold standard for user simulation (1C) calibration |
| 2 | HeartSteps V2 | ~97 | Fitbit (steps, HR, sleep) | 90 days | Restricted (contact lab) | — | S, H, L, T, I, A | RL-in-the-loop benchmark (1D), Thompson Sampling |
| 3 | StepCountJITAI | ~50 | Simulated from MRT | Variable | Academic use | PR #59 | S, T, I, A | RL simulation environment for PA JITAIs |
| 4 | 4TU Step Goals | ~100 | Step count | Variable | Open (CC BY 4.0) | PR #60 | S, G, T, I | RL-based step goal personalization |
| 5 | TILES | ~200 | Fitbit + smartphone | 12 months | Open (OSF) | — | S, H, L, E, T, I, A | Longitudinal intervention response, behavioral drift |
| | | | | | | | | |
| | **TIER 2: LARGE-SCALE WEARABLE** | | | | | | | |
| 6 | All of Us Fitbit | 59,018 | Fitbit (steps, HR, sleep, calories) | 14 years | Institutional (DURA, 4-8wk) | PR #70 | S, H, L, E, T, D | Population distributions, synthetic parameterization |
| 7 | UK Biobank Accel | 100,000+ | Tri-axial accel (100Hz) | 7 days | Institutional (UKB app, 4-8wk, £500-9000) | — | S, E, T, D | Sedentary behaviour, circadian analysis |
| 8 | NHANES Steps | 14,693 | ActiGraph GT3X+ (80Hz) | 7 days | Open (CC0, PhysioNet) | — | S, D | Step distribution parameterization (largest open) |
| | | | | | | | | |
| | **TIER 3: OPEN WEARABLE + SLEEP** | | | | | | | |
| 9 | DREAMT | 100 | Empatica E4 + PSG | Multi-night | Open (PhysioNet) | PR #68 | H, L | Wearable + gold-standard sleep staging |
| 10 | BIDSleep | ~60 | Apple Watch + Dreem 2 EEG | Multi-night | Open (PhysioNet) | PR #65 | H, L | Consumer wearable HR/accel with sleep labels |
| 11 | GLOBEM | 497 | Fitbit + phone (4 years) | 10 wks/yr | Credential (PhysioNet) | — | S, L, D | Multi-year behavioral patterns, non-stationarity |
| 12 | LifeSnaps | 71 | Fitbit Charge 2 + phone + EMA | 4 weeks | Open (4TU) | — | S, H, L, D | EMA-coupled wearable data, activity-mood coupling |
| 13 | MMASH | 22 | Polar H7 + ActiGraph | 24 hours | Open (PhysioNet, ODbL) | — | S, H, L | Multimodal wearable (HR, accel, cortisol, melatonin) |
| | | | | | | | | |
| | **TIER 4: ACTIVITY RECOGNITION** | | | | | | | |
| 14 | HARTH Adults | 31 | Dual AX3 accel (thigh+back, 50Hz) | Variable | Open (MIT, HuggingFace) | PR #61 | S, E, T, D | 12 activity types, body-worn sensors |
| 15 | HARTH Children | Variable | Dual AX3 accel | Variable | Open (MIT, HuggingFace) | PR #61 | S, E, T, D | Same as adults, child population |
| 16 | HARTH Older Adults | Variable | Dual AX3 accel | Variable | Open (MIT, HuggingFace) | PR #61 | S, E, T, D | Same as adults, older population |
| 17 | HARTH Wrist | Variable | AX3 accel (wrist) | Variable | Open (MIT, HuggingFace) | PR #61 | S, E, T, D | Wrist-only variant |
| 18 | PAMAP2 | 9 | 3x IMU + HR monitor | ~1.5 hrs each | Open (UCI, HuggingFace) | — | S, H | 3.85M labelled data points, 12 activities |
| 19 | WISDM | 36 | Phone accelerometer (20Hz) | Variable | Open (Fordham) | Config exists | S, T | 6 activities, pipeline testing |
| 20 | MHEALTH | 10 | Chest (ECG, accel, gyro, mag) + wrist + ankle | Variable | Open (CC BY 4.0, UCI) | PR #63 | S, H, E, T, D | Multi-body sensor, ECG validates HR |
| 21 | ExtraSensory | 60 | Phone (accel, gyro, compass, audio, GPS) + smartwatch | ~7 days | Open (UCSD) | Config exists | S, H, T, D | Multi-sensor pipeline testing |
| 22 | MyHeartCounts | ~48,000 | Apple Watch | 7 days | Restricted (Stanford DUA) | — | S, H, L | Large-scale Apple Watch data |
| | | | | | | | | |
| | **TIER 5: STRESS / AFFECT / BIOMETRIC** | | | | | | | |
| 23 | WESAD | 15 | Chest (ECG, EDA, EMG, resp, temp) + wrist (accel, EDA, BVP, temp) | Variable | Academic use (Kaggle) | PR #66 | B, A | Stress/affect labels for burden modelling |
| 24 | PMData | Variable | Garmin + WHOOP + Apple + HRV4Training | Variable | Open (CC BY 4.0, Kaggle) | PR #64 | S, H, L, D | 1GB multi-source, HRV, recovery, training load |
| 25 | Fitbit Tracker (Kaggle) | 33 | Fitbit (steps, HR, sleep, calories) | Variable | Open (CC0, Kaggle) | PR #62 | S, H, L, E | Real Fitbit data, daily/minute level |
| 26 | ScientISST MOVE | Variable | Wearable multimodal biosignals | Variable | Open (PhysioNet) | PR #67 | S, H, E, T | Everyday life activities, ecological validity |
| 27 | Wearable Device (Exercise) | Variable | Wrist-worn wearable | Variable | Open (PhysioNet) | — | S, H | Stress + exercise sessions |
| 28 | BIG IDEAs Glycemic | Variable | CGM + wearable | Variable | Open (PhysioNet) | — | S | CGM + physical activity |
| | | | | | | | | |
| | **TIER 6: SYNTHETIC / RL ENVIRONMENTS** | | | | | | | |
| 29 | Health Gym | N/A | Synthetic from MIMIC-III | N/A | Open (MIT, GitHub) | — | — | HIV, Sepsis, Acute Hypotension RL environments |
| 30 | Synthetic (our own) | Configurable | Configurable | Configurable | Generated | Config exists | All | Configurable synthetic data generator |
| | | | | | | | | |
| | **TIER 7: CLINICAL / NICHE** | | | | | | | |
| 31 | LiveWell | 10 | Wearable camera + Fitbit + GPS | 4 months | Restricted (contact lab) | — | S, H, E, T | Wearable cameras, eating/PA monitoring |
| 32 | CovIdentify | Variable | Multiple wearables (Fitbit, Apple Watch) | Variable | Open (PhysioNet) | — | S, H | COVID symptom tracking, longitudinal wearable |
| 33 | mcPHASES | Variable | Wearable | Variable | Open (PhysioNet) | — | H, L | Menstrual health, longitudinal wearable |
| 34 | Sleep-EDF | 82 | PSG (EEG, EOG, EMG) | Nightly | Open (PhysioNet) | — | L | Classic sleep staging benchmark |
| 35 | NCH Sleep DataBank | 3,000+ | PSG | Variable | Open (PhysioNet) | — | L | Pediatric sleep, longitudinal clinical |

---

## Access Summary

| Access Level | Count | Datasets |
|-------------|-------|----------|
| **Open (download now)** | 16 | NHANES Steps, DREAMT, BIDSleep, HARTH (x4), PAMAP2, WISDM, ExtraSensory, MHEALTH, WESAD, PMData, Fitbit Tracker, ScientISST MOVE, LifeSnaps, Health Gym |
| **Open (credential required)** | 4 | 4TU Step Goals, GLOBEM, MMASH, UK Biobank Accel |
| **Institutional (DURA, 4-8wk)** | 2 | All of Us Fitbit, MyHeartCounts |
| **Restricted (contact lab)** | 3 | HeartSteps V1/V2, TILES, LiveWell |
| **Academic use** | 1 | StepCountJITAI |

---

## MDP Coverage Matrix

| Dataset | S | H | L | E | T | D | G | B | A | I | K |
|---------|---|---|---|---|---|---|---|---|---|---|---|
| HeartSteps V2 | ✓ | ✓ | ✓ | — | ✓ | ✓ | — | — | ✓ | ✓ | — |
| TILES | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | ✓ | — |
| All of Us | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — | — | ✓* |
| 4TU Step Goals | ✓ | — | — | — | ✓ | — | ✓ | — | ✓ | ✓ | — |
| DREAMT | — | ✓ | ✓ | — | — | — | — | — | — | — | — |
| BIDSleep | — | ✓ | ✓ | — | — | — | — | — | — | — | — |
| HARTH | ✓ | — | — | ✓ | ✓ | ✓ | — | — | — | — | — |
| PMData | ✓ | ✓ | ✓ | — | — | ✓ | — | — | — | — | — |
| WESAD | — | — | — | — | — | — | — | ✓ | ✓ | — | — |
| MHEALTH | ✓ | ✓ | — | ✓ | ✓ | ✓ | — | — | — | — | — |

*46% EHR linkage coverage

---

## Recommendation: What to Start Downloading Now

**Immediate (no application needed):**

| Priority | Dataset | Why | Effort |
|----------|---------|-----|--------|
| 1 | NHANES Steps (CC0) | 14.7K participants, step distributions, population baseline | Low |
| 2 | HARTH (MIT) | 12 activity types, 4 demographics, body-worn sensors | Low |
| 3 | Fitbit Tracker (CC0) | Real Fitbit data, steps/HR/sleep | Low |
| 4 | PAMAP2 | Multi-sensor, 12 activities, 3.85M labelled points | Low |
| 5 | MHEALTH | Chest ECG + wrist + ankle, validates HR | Low |
| 6 | MMASH | Multimodal wearable + cortisol/melatonin | Low |

**Start application now (4-8 week lead time):**

| Priority | Dataset | Why | Effort |
|----------|---------|-----|--------|
| 1 | All of Us Fitbit | Richest population dataset, 59K participants | High (cloud-only) |
| 2 | UK Biobank | 100K+ participants, accelerometry | High (institutional) |
| 3 | HeartSteps V1/V2 | Only gold-standard intervention response data | Medium (contact lab) |

**Download and run (when available):**

| Priority | Dataset | Why | Effort |
|----------|---------|-----|--------|
| 1 | 4TU Step Goals | RL intervention data, CC BY 4.0 | Low |
| 2 | TILES | 12-month longitudinal intervention | Medium |
| 3 | StepCountJITAI | Purpose-built RL sim for PA JITAIs | Medium |
