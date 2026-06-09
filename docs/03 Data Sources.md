# Data Sources

This document describes the public wearable health datasets available for training the rule-based user simulator.

## Overview

| Dataset | Participants | Data Type | Access | Format |
|---------|-------------|-----------|--------|--------|
| All of Us Fitbit | 59,000+ | Fitbit (steps, HR, sleep) | Controlled | CSV |
| UK Biobank Accelerometer | 100,000+ | Wrist accelerometer | Controlled | CSV |

## 1. All of Us Fitbit Dataset

**Source:** [Nature Medicine 2026](https://www.nature.com/articles/s41591-026-04352-3)

### Access

1. Register at [researchallofus.org](https://researchallofus.org/)
2. Complete required training (CITI + Data or Specimens Only)
3. Submit a research application
4. Access data via the All of Us Researcher Workbench (cloud-based)

**Note:** Data cannot be downloaded locally. Analysis runs in the cloud workbench.

### Schema

| Field | Column | Type | Unit |
|-------|--------|------|------|
| Participant ID | `person_id` | string | — |
| Date | `date` | datetime | — |
| Steps | `steps` | integer | count |
| Resting Heart Rate | `resting_heart_rate` | float | bpm |
| Heart Rate Minutes | `heart_rate_minutes` | float | minutes |
| Sleep Minutes | `sleep_minutes` | float | minutes |
| Sleep Efficiency | `sleep_efficiency` | float | ratio (0-1) |
| Calories | `calories` | float | kcal |
| Distance | `distance_meters` | float | meters |
| Floors Climbed | `floors_climbed` | integer | count |

### Semantic Mapping

| MDP Field | Dataset Column |
|-----------|---------------|
| steps | `steps` |
| heart_rate | `resting_heart_rate` |
| sleep | `sleep_minutes` |
| time_of_day | `date` |

### Usage Notes

- 39M+ step observations, 31M+ sleep observations
- 46% linked to EHR, physical measurements, genomics
- 14-year span (longitudinal)
- Fitbit data includes minute-level granularity
- Resting heart rate available for most participants

### Config

```yaml
# configs/datasets/all_of_us_fitbit.yaml
dataset:
  name: all_of_us_fitbit
  access: controlled_access
```

## 2. UK Biobank Accelerometer Dataset

**Source:** [npj Digital Medicine 2024](https://www.nature.com/articles/s41746-024-01062-3)
**Code:** [OxWearables/ssl-wearables](https://github.com/OxWearables/ssl-wearables)

### Access

1. Register at [ukbiobank.ac.uk](https://www.ukbiobank.ac.uk/register-apply/)
2. Submit a Research Application (requires institutional affiliation)
3. Pay application fee (~£500-9000 depending on scope)
4. Data delivered via UK Biobank Research Analysis Platform

### Schema

| Field | Column | Type | Unit |
|-------|--------|------|------|
| Participant ID | `eid` | string | — |
| Date | `date` | datetime | — |
| Steps | `steps` | integer | count |
| Wear Time | `wear_time` | float | minutes |
| Accelerometer X | `x` | float | g |
| Accelerometer Y | `y` | float | g |
| Accelerometer Z | `z` | float | g |
| Angle Z | `anglez` | float | degrees |
| ENMO | `enmo` | float | mg |
| Pitch | `pitch` | float | degrees |
| Roll | `roll` | float | degrees |
| Light | `light` | float | lux |
| Temperature | `temperature` | float | celsius |

### Semantic Mapping

| MDP Field | Dataset Column |
|-----------|---------------|
| steps | `steps` |
| heart_rate | *not available* |
| sleep | `wear_time` |
| time_of_day | `date` |
| accelerometer | `enmo` |

### Usage Notes

- 700,000+ person-days of data
- 100,000+ participants, 7 days each
- Raw tri-axial accelerometer at 100Hz
- Derived features (ENMO, steps) computed from raw
- No heart rate data — use All of Us for HR
- OxWearables provides pre-trained SSL models for sleep/wake detection
- ENMO (Euclidean Norm Minus One) is the standard movement intensity metric

### Config

```yaml
# configs/datasets/uk_biobank_accelerometer.yaml
dataset:
  name: uk_biobank_accelerometer
  access: controlled_access
```

## Dataset Selection Guide

| Need | Use |
|------|-----|
| Heart rate data | All of Us |
| High-frequency movement | UK Biobank (100Hz) |
| Longitudinal (years) | All of Us (14 years) |
| Large N, short duration | UK Biobank (7 days, 100K people) |
| EHR linkage | All of Us (46% linked) |
| Sleep staging | UK Biobank (OxWearables models) |
| Both heart rate + steps | All of Us |

## Using the Data Layer

### Load a dataset

```python
from rl_health_interventions.data import load_dataset

df = load_dataset(
    config_path="configs/datasets/all_of_us_fitbit.yaml",
    data_dir="/path/to/data/",
)
# df has columns: steps, heart_rate, sleep, time_of_day, ...
```

### Preprocess

```python
from rl_health_interventions.data.preprocessing import (
    resample_to_hourly,
    compute_daily_aggregates,
    add_time_features,
)

# Resample to hourly
hourly = resample_to_hourly(df, time_col="time_of_day")

# Daily aggregates
daily = compute_daily_aggregates(df, group_col="participant_id")

# Add temporal features
df = add_time_features(df, time_col="time_of_day")
```

### Add a new dataset

1. Create `configs/datasets/your_dataset.yaml` following the schema format
2. Ensure the `semantic_mapping` covers at least `steps`
3. Place CSV files in your data directory
4. Load with `load_dataset(config_path, data_dir)`
