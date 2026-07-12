# All of Us Fitbit Dataset

**Date:** 2026-06-10
**Context:** RL Health Interventions — Dataset Discovery

---

## Overview

| Property | Value |
|----------|-------|
| Paper | Patten et al. (2026), Nature Medicine |
| URL | https://www.nature.com/articles/s41591-026-04352-3 |
| Participants | 59,000+ |
| Data type | Fitbit (steps, heart rate, sleep, calories, distance) |
| Timespan | 14 years (longitudinal) |
| Step observations | 39M+ |
| Sleep observations | 31M+ |
| HR observations | Minute-level and daily summaries |
| EHR linkage | 46% linked to EHR, physical measurements, genomics |
| Format | CSV (via cloud workbench only) |
| Relevance | **HIGHEST** — richest single source for wearable state variables |

## Data Tables (from AoU Researcher Workbench)

The dataset is stored as BigQuery tables in the AoU Controlled Tier. Key tables:

### `activity_summary` — Daily activity aggregates
| Column | Type | Description |
|--------|------|-------------|
| person_id | STRING | Participant identifier |
| date | DATE | Observation date |
| steps | INT64 | Daily step count |
| calories | FLOAT64 | Calories burned |
| distance_meters | FLOAT64 | Distance in meters |
| floors_climbed | INT64 | Floors climbed |

### `steps_intraday` — Minute-level step counts
| Column | Type | Description |
|--------|------|-------------|
| person_id | STRING | Participant identifier |
| datetime | TIMESTAMP | Minute-level timestamp |
| steps | INT64 | Steps in that minute |

### `sleep_daily_summary` — Daily sleep summary
| Column | Type | Description |
|--------|------|-------------|
| person_id | STRING | Participant identifier |
| sleep_date | DATE | Sleep date |
| is_main_sleep | BOOLEAN | Main sleep vs nap |
| minute_asleep | INT64 | Minutes asleep |

### `sleep_level` — Sleep stage data
| Column | Type | Description |
|--------|------|-------------|
| person_id | STRING | Participant identifier |
| sleep_date | DATE | Sleep date |
| sleep_level | STRING | Sleep stage (light, deep, rem, wake) |

### `heart_rate_summary` — Daily HR summary
| Column | Type | Description |
|--------|------|-------------|
| person_id | STRING | Participant identifier |
| date | DATE | Observation date |
| resting_heart_rate | FLOAT64 | Resting HR (bpm) |
| heart_rate_minutes | FLOAT64 | Minutes with HR data |

### `heart_rate_minute_level` — Minute-level HR
| Column | Type | Description |
|--------|------|-------------|
| person_id | STRING | Participant identifier |
| datetime | TIMESTAMP | Minute-level timestamp |
| heart_rate | FLOAT64 | HR (bpm) |

### `person` — Demographics
| Column | Type | Description |
|--------|------|-------------|
| person_id | STRING | Participant identifier |
| birth_datetime | TIMESTAMP | Date of birth (for age calculation) |

### `device` — Device metadata
| Column | Type | Description |
|--------|------|-------------|
| person_id | STRING | Participant identifier |
| device_type | STRING | Device type (Fitbit model) |

### Derived: `valid_day` — Compliance / wear time
Computed from `steps_intraday`. A valid day = 10+ hours with non-zero steps.

| Column | Type | Description |
|--------|------|-------------|
| person_id | STRING | Participant identifier |
| date | DATE | Observation date |
| hours_worn | INT64 | Hours with step data |
| compliant | BOOLEAN | 10+ hours worn |

## MDP Variable Mapping

| MDP Variable | Available? | Source | Notes |
|---|---|---|---|
| `steps_t` | Direct | activity_summary.steps | Daily step counts |
| `hr_t` | Direct | heart_rate_summary.resting_heart_rate | Daily resting HR |
| `sleep_hours_t` | Direct | sleep_daily_summary.minute_asleep / 60 | Convert minutes to hours |
| `sedentary_min_t` | Derivable | steps_intraday (0-step minutes today) | Inverse of active minutes |
| `time_of_day_t` | Derivable | steps_intraday.datetime | Extract hour |
| `day_of_week_t` | Derivable | activity_summary.date | Extract weekday |
| `age` | Direct | person.birth_datetime | Calculate from date of birth |
| `gender` | Direct | ds_survey (question_concept_id=1585845) | Sex at birth |
| `baseline_activity` | Derived | Step distributions per user | Low/medium/high activity level |
| `body_measure_k` | Via EHR link | 46% of participants | Physical measurements, clinical |
| `goal_progress_t` | Synthetic | Not in data | Must simulate |
| `burden_t` | Synthetic | Not in data | Must simulate |
| `previous_action` | Synthetic | Not in data | Must simulate |
| `previous_response` | Synthetic | Not in data | Must simulate |

## Key Limitations

1. **Observational only** — No intervention delivery or user response data. Cannot calibrate user simulation (1C) action-response dynamics.
2. **Cloud-only** — Data cannot leave the Researcher Workbench. All analysis runs in their cloud environment.
3. **No downloadable samples** — Must apply for access to inspect any data.
4. **No minute-level heart rate for all** — HR coverage varies by participant.

## Access Requirements

1. Register at https://researchallofus.org/
2. Complete CITI training + Data or Specimens Only certification
3. Obtain Data Use and Registration Agreement (DURA) from your institution
4. Submit research application (4-8 week review)
5. Access via https://workbench.researchallofus.org

## Accessing the Data (SQL Reference)

From the AoU GitHub repo (RTIInternational/allofus_NIH_wear):

```sql
-- Daily steps
SELECT person_id, date, steps
FROM `dataset.activity_summary`
WHERE steps > 0

-- Minute-level steps
SELECT person_id, datetime, steps
FROM `dataset.steps_intraday`

-- Sleep summary
SELECT person_id, sleep_date, is_main_sleep, minute_asleep
FROM `dataset.sleep_daily_summary`

-- Heart rate (minute-level)
SELECT person_id, datetime, heart_rate
FROM `dataset.heart_rate_minute_level`

-- Demographics + age
SELECT person_id, birth_datetime
FROM `dataset.person`

-- Valid day (compliant = 10+ hours with steps)
SELECT
  person_id,
  CAST(datetime AS DATE) AS date,
  COUNT(DISTINCT EXTRACT(HOUR FROM datetime)) AS hours_worn,
  COUNT(DISTINCT EXTRACT(HOUR FROM datetime)) >= 10 AS compliant
FROM `dataset.steps_intraday`
WHERE steps > 0
GROUP BY person_id, date
```

## Files Reference

| File | Purpose |
|------|---------|
| `sources/data-sources.md` | Original feasibility study (covers AoU + UK Biobank) |
| `initial_design.tex` | Formal MDP definition with state variables |
| `config/datasets/allofus_fitbit.yaml` | Data ingestion config (for when access is granted) |
