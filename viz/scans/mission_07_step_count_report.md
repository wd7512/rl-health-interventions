# Mission 07: Step Count Modeling & Biomechanics (2020-2026)

## Mission Overview

- **Mission ID:** 07
- **Topic:** Step Count Modeling & Biomechanics
- **Search Period:** 2020-2026
- **Target Paper Count:** 15-25
- **Papers Collected:** 25
- **Date:** 2026-07-13

## Search Strategy

Searched Europe PMC, PubMed, and arXiv using the following query themes:

1. `step detection algorithm validation wearable` (36 results)
2. `activity recognition wearable machine learning step count` (4 results)
3. `step count machine learning wearable accuracy` (broad results)
4. `demographic bias inequity step count wearable` (84 results)
5. `device comparison step count accuracy` (24,700 results)
6. `energy expenditure wearable machine learning` (1,955 results)
7. `ground truth step validation accelerometer walking` (1,029 results)
8. `step count clinical population parkinson stroke slow walking` (392 results)
9. `smartwatch step count accuracy underestimation overestimation` (48 results)
10. `deep learning step detection inertial measurement unit` (1,494 results)

## Paper Summary Table

| ID | Year | Category | Population | Device | Key Finding |
|----|------|----------|------------|--------|-------------|
| SC-01 | 2026 | Step Detection Algorithm | Healthy adults | IMU | Peak detection best for edge; speed affects tradeoff |
| SC-02 | 2021 | Step Detection Algorithm | Adults | ActiGraph | Transparent threshold method matches commercial algos |
| SC-03 | 2023 | Step Detection Algorithm | Cancer patients + healthy | Smartphone | Open-source step algorithm viable in oncology trials |
| SC-04 | 2024 | Step Detection Algorithm | Older adults + impaired | Wrist IMU (AX3) | Gait detection feasible but arm movement complexity matters |
| SC-05 | 2026 | Activity Recognition | Healthy adults | Smartwatch IMU | Ultra-lightweight CNN for edge HAR deployment |
| SC-06 | 2026 | Activity Recognition | Healthy adults | Research accelerometer | Open dataset enables reproducible HAR benchmarking |
| SC-07 | 2026 | Activity Recognition | Older adults | Wrist AX3 | Self-supervised gait assessment without manual labels |
| SC-08 | 2026 | Activity Recognition | Young adults | Waist IMU | Combined HAR+EE framework >90% accuracy |
| SC-09 | 2026 | Step Count ML | Adults | Multi-wearable | ML outperforms thresholds; step count alone insufficient |
| SC-10 | 2026 | Step Count ML | Warehouse workers | IMU+location | Multimodal fusion beats step-only for context-rich HAR |
| SC-11 | 2026 | Device Comparison | Older adults (mobility aids) | Smartwatch | **Negative:** Accuracy degrades with walkers/canes |
| SC-12 | 2026 | Device Comparison | Healthy adults | Consumer vs ActiGraph | Variable agreement; device-specific biases |
| SC-13 | 2025 | Device Comparison | Simulated impaired gait | Apple Watch, Fitbit, Garmin | Apple Watch best for slow walking; all fail below 0.8 m/s |
| SC-14 | 2024 | Device Comparison | Wheelchair users | Apple Watch, Fitbit | **Negative:** MAPE >30% for EE in wheelchair propulsion |
| SC-15 | 2020 | Device Comparison | Healthy adults | Low-cost trackers | Group-level acceptable; individual-level poor |
| SC-16 | 2025 | Clinical Accuracy | CVD, PAD patients | Life Plus Watch | Speed-dependent errors; clinical populations differ |
| SC-17 | 2022 | Clinical Accuracy | Parkinson's disease | Smartwatch | Underestimates steps; OFF medication worse |
| SC-18 | 2024 | Clinical Accuracy | Healthy adults | Huawei GT2 | Acceptable at normal speeds; underestimates slow walking |
| SC-19 | 2026 | Energy Expenditure | Young adults | ActTrust | Group-level EE OK; individual errors substantial |
| SC-20 | 2026 | Energy Expenditure | Healthy adults | Smartphone (pocket) | OpenMetabolics matches research accelerometers |
| SC-21 | 2025 | Energy Expenditure | Healthy adults | Sacrum IMU | Biomechanical features improve ML-based EE estimation |
| SC-22 | 2026 | Demographic Bias | General (review) | Multiple | Measurement error differs by race, BMI, age |
| SC-23 | 2026 | Demographic Bias | General (review) | Multiple | Digital health excludes older, lower SES, minorities |
| SC-24 | 2025 | Ground Truth | Adults (free-living) | Multi-reference | Comprehensive protocol for free-living validation |
| SC-25 | 2025 | Ground Truth | ALS patients | Smartphone/accel | Lower accuracy in neurodegenerative population |

## Key Themes & Findings

### 1. Step Detection Algorithms (SC-01 through SC-04)

The field continues to debate transparent vs. proprietary algorithms. Ducharme et al. (2021) demonstrated that simple acceleration threshold methods can match commercial algorithm performance, providing a reproducible research alternative. Kisiel et al. (2026) found peak detection optimal for resource-constrained edge deployment. Straczkiewicz et al. (2023) validated an open-source smartphone algorithm in oncology trials. Kluge et al. (2024) showed wrist-worn gait detection is feasible but arm movement complexity introduces error.

**Implication for JITAI state:** Algorithm choice systematically affects step_bin assignment, especially at boundary walking speeds. An open, validated algorithm should be preferred over proprietary SDKs.

### 2. Activity Recognition with ML (SC-05 through SC-08)

Ultra-lightweight deep learning models (Xu et al. 2026) now enable on-device HAR. Self-supervised learning (Brand et al. 2026) eliminates the need for labeled training data in everyday gait assessment. Combined frameworks (Zhou et al. 2026) integrate activity recognition with EE classification from a single IMU. Open datasets (Nikookar et al. 2026) enable reproducible benchmarks.

**Implication for JITAI state:** ML-based activity classification can supplement or replace simple step count binned states, potentially offering richer context for intervention timing.

### 3. Device Comparison Failures (SC-11 through SC-15)

Multiple studies document significant accuracy failures:
- **Mobility aids:** Smartwatch accuracy degrades with walkers/canes (Ozel et al. 2026)
- **Wheelchair users:** Both Apple Watch and Fitbit show MAPE >30% for EE (Danielsson et al. 2024)
- **Slow walking:** All devices underestimate steps below 0.8 m/s (Rowe et al. 2025)
- **Low-cost trackers:** Group-level OK but individual-level precision poor (Degroote et al. 2020)

### 4. Clinical Population Accuracy (SC-16 through SC-18)

Smartwatch accuracy varies systematically across clinical populations:
- CVD/PAD patients show different error patterns than healthy (Heizmann et al. 2025)
- PD patients: underestimation worsens in OFF medication state and with more severe symptoms (Bianchini et al. 2022)
- ALS patients: lower accuracy at slower speeds (Straczkiewicz et al. 2025)

**Implication for JITAI state:** Demographic and clinical factors systematically modify the mapping from sensor signal to step_bin. A model that treats step_bin as a simple threshold of raw accelerometer data may encode systematic biases.

### 5. Demographic Bias & Equity (SC-22, SC-23)

Mai et al. (2026) document that wearable measurement error differs systematically by race, ethnicity, BMI, and age. Jiang et al. (2026) find that digital health interventions systematically exclude older adults, lower SES groups, and ethnic minorities. Without correction, these biases propagate through JITAI models, potentially widening health disparities.

### 6. Energy Expenditure (SC-19 through SC-21)

EE estimation from wearables is improving but individual-level errors remain substantial. Biomechanically-informed ML (Jung et al. 2025) and smartphone-based methods (Cho & Slade 2026) show promise for scalable population-level monitoring, but clinical-grade precision remains elusive.

### 7. Ground Truth Validation (SC-24, SC-25)

The OxWEARS protocol (Maylor et al. 2025) represents a gold-standard approach combining chest ECG, body camera, ankle accelerometer, and polysomnography. ALS validation (Straczkiewicz et al. 2025) demonstrates the challenges of step counting in neurodegenerative populations.

## Research Gaps Identified

1. **Proprietary algorithm opacity:** Most consumer devices do not disclose step detection algorithms, making reproducibility impossible.
2. **Diverse population underrepresentation:** The majority of step count validation studies use healthy, young-to-middle-aged adult samples.
3. **Free-living validation scarcity:** Most studies are laboratory-based; free-living ground truth remains expensive and rare.
4. **Bias correction methods absent:** Despite evidence of demographic bias, few studies propose correction methods for step count models.
5. **Edge case underdocumentation:** Slow walking (<0.8 m/s), assistive device use, wheelchair propulsion, and non-steady-state activities are poorly characterized.
6. **Temporal resolution mismatch:** Most step count output is daily aggregates, but JITAI systems require sub-minute or minute-level step_bin states.

## Recommendations for Config-Driven RL JITAI

1. **Use transparent, validated algorithms** (e.g., Ducharme 2021, Straczkiewicz 2023) rather than proprietary SDKs for step detection.
2. **Calibrate step_bin thresholds** specifically for the target population's gait characteristics (speed, assistive devices, clinical status).
3. **Account for systematic bias** by collecting demographic-stratified validation data.
4. **Consider ML-based activity classification** as a supplement or alternative to simple step_bin for state representation.
5. **Validate at sub-minute resolution** to ensure step_bin transitions are temporally aligned with intervention opportunities.
6. **Document sensor placement and device** explicitly, as these factors substantially affect measurement accuracy.
