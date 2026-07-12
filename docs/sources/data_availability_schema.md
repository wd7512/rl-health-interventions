---
title: "Data Availability & Schema Reference"
status: "reference"
last_reviewed: "2026-07-12"
---

# Data Availability & Schema Reference

## Overview

| # | Dataset | Status | Rows | Access | License |
|---|---------|--------|------|--------|---------|
| 1 | PMData | **YES** | 1,836 | Kaggle API | CC BY 4.0 |
| 2 | HARTH | **YES** | 8,322,728 | HuggingFace Datasets | MIT |
| 3 | WESAD | **YES** | 60,807,600 | Kaggle API | Academic |
| 4 | WISDM | **YES** | 1,098,209 | Direct download (tar.gz) | Public domain |
| 5 | Fitbit Tracker | **YES** | 940 | Kaggle API | CC0 |
| 6 | 4TU Step Goals | **YES** | 117 | Direct download (ZIP) | CC BY 4.0 |
| 7 | Synthetic | **YES** | 36,500 | Local generation | Public domain |
| 8 | BIDSleep | **NO** | — | PhysioNet (wfdb) | Open access |
| 9 | ScientISST MOVE | **NO** | — | PhysioNet (wfdb) | Open access |
| 10 | DREAMT | **NO** | — | PhysioNet (wfdb) | Open access |
| 11 | MHEALTH | **NO** | — | UCI ML Repo (ucimlrepo) | CC BY 4.0 |
| 12 | ExtraSensory | **NO** | — | Direct download (ucsd.edu) | Public domain |
| 13 | All of Us Fitbit | **NO** | 39M+ | AoU Researcher Workbench (BigQuery) | Controlled tier |
| 14 | StepCountJITAI | **NO** | — | Restricted academic access | Academic |

---

## 1. PMData

- **Status:** YES
- **Access:** `kagglehub.dataset_download("vlbthambawita/pmdata-a-sports-logging-dataset")`
- **Credentials:** Kaggle API token (`~/.kaggle/kaggle.json`)
- **File format:** CSV (one `sleep_score.csv` per participant, 16 participants)
- **Size:** 1,836 rows, ~11 columns

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Participant ID (p01–p16), extracted from directory name |
| `timestamp` | datetime | Date of measurement (original column: `date`) |
| `heart_rate` | float | Resting/average heart rate |
| `heart_rate_variability` | float | HRV metric |
| `step_count` | float | Daily step count |
| `sleep_duration` | float | Sleep duration (hours or minutes) |
| `sleep_stages` | string | Sleep stage breakdown (e.g., "deep 1.5h, light 4h") |
| `recovery_score` | float | Garmin body battery / recovery score |
| `spo2` | float | Blood oxygen saturation |
| `body_temperature` | float | Skin/body temperature |
| `stress_level` | float | Stress level score |

---

## 2. HARTH

- **Status:** YES
- **Access:** `datasets.load_dataset("josefheidler/har_adults_2021-harth")`
- **Credentials:** None (HuggingFace public dataset)
- **File format:** HuggingFace Datasets (Apache Arrow)
- **Size:** 8,322,728 rows, 9 columns

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Subject ID (original column: `subject`) |
| `timestamp` | datetime | Observation timestamp |
| `thigh_accel_x` | float | Thigh accelerometer X-axis (50 Hz) |
| `thigh_accel_y` | float | Thigh accelerometer Y-axis |
| `thigh_accel_z` | float | Thigh accelerometer Z-axis |
| `back_accel_x` | float | Back accelerometer X-axis |
| `back_accel_y` | float | Back accelerometer Y-axis |
| `back_accel_z` | float | Back accelerometer Z-axis |
| `activity` | string | Activity label (12 types) |

---

## 3. WESAD

- **Status:** YES
- **Access:** `kagglehub.dataset_download("orvile/wesad-wearable-stress-affect-detection-dataset")`
- **Credentials:** Kaggle API token
- **File format:** Pickle (`.pkl`), one per subject
- **Size:** 60,807,600 rows, 11 columns

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Subject ID (S2, S3, S5, etc.) |
| `timestamp` | int64 | Sample index (sequential) |
| `chest_acc_x` | float | Chest accelerometer X-axis |
| `chest_acc_y` | float | Chest accelerometer Y-axis |
| `chest_acc_z` | float | Chest accelerometer Z-axis |
| `chest_ecg` | float | Chest ECG signal |
| `chest_eda` | float | Chest EDA (electrodermal activity) |
| `chest_emg` | float | Chest EMG |
| `chest_resp` | float | Chest respiration signal |
| `chest_temp` | float | Chest temperature |
| `label` | string | Condition (baseline, stress, amusement, meditation) |

> **Note:** The loader extracts chest-worn sensors only. Wrist signals (BVP, EDA, Temp, Accel) are in the raw pickle files but not returned.

---

## 4. WISDM

- **Status:** YES
- **Access:** Direct download of `WISDM_ar_latest.tar.gz` from `https://www.cis.fordham.edu/wisdm/includes/datasets/latest/`
- **Credentials:** None
- **File format:** CSV (no header, semicolons stripped)
- **Size:** 1,098,209 rows, 6 columns

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Subject ID |
| `timestamp` | datetime | Unix nanosecond epoch |
| `activity` | string | Activity label (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing) |
| `accelerometer_x` | float | Phone accelerometer X-axis |
| `accelerometer_y` | float | Phone accelerometer Y-axis |
| `accelerometer_z` | float | Phone accelerometer Z-axis |

---

## 5. Fitbit Tracker

- **Status:** YES
- **Access:** `kagglehub.dataset_download("haseeb85/fitbit-fitness-tracker-data")`
- **Credentials:** Kaggle API token
- **File format:** CSV (`dailyActivity_merged.csv`)
- **Size:** 940 rows, 10 columns

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | User ID (original column: `Id`) |
| `timestamp` | datetime | Activity date (original column: `ActivityDate`) |
| `step_count` | float | Daily step total |
| `heart_rate` | float | Heart rate (sparse) |
| `sleep_minutes` | float | Minutes asleep |
| `sleep_efficiency` | float | Sleep efficiency |
| `calories` | float | Calories burned |
| `distance` | float | Distance traveled |
| `active_minutes` | float | Active wear minutes |
| `sedentary_minutes` | float | Sedentary minutes |

---

## 6. 4TU Step Goals

- **Status:** YES
- **Access:** Direct download from `https://data.4tu.nl/file/...`
- **Credentials:** None
- **File format:** CSV (inside ZIP)
- **Size:** 117 rows, 7 columns

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Participant ID |
| `timestamp` | datetime | Date of observation |
| `step_count` | float | Daily step count |
| `assigned_goal` | float | RL-assigned step goal |
| `goal_met` | boolean | Whether goal was achieved |
| `intervention_type` | string | Intervention type (goal, feedback, etc.) |
| `day_of_week` | integer | Day of week (0=Monday, 6=Sunday) |

---

## 7. Synthetic

- **Status:** YES
- **Access:** Local generation via `SyntheticDataGenerator` (no download)
- **Credentials:** None
- **File format:** Generated in-memory (Polars DataFrame)
- **Size:** 36,500 rows (100 users x 365 days), 3 columns

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | User ID (0–99) |
| `timestep` | int64 | Day index (0–364) |
| `steps` | int64 | Synthetic step count ~N(8000, 2000) |

> The YAML config defines a broader schema (`timestamp`, `activity`, `accelerometer_x/y/z`), but the actual loader produces only `user_id`, `timestep`, `steps`.

---

## 8. BIDSleep

- **Status:** NO — requires PhysioNet credentials + accepted terms of use
- **Access:** `wfdb.dl_database("bidsleep-dataset/1.0.0", ...)`
- **Credentials:** `PHYSIONET_USERNAME` + `PHYSIONET_PASSWORD` (or `WFDB_USERNAME` + `WFDB_PASSWORD`)
- **File format:** BIDS (TSV/CSV)
- **License:** Open access

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Participant ID |
| `night` | integer | Recording night number |
| `timestamp` | datetime | Measurement timestamp |
| `heart_rate` | float | Heart rate from Apple Watch |
| `accel_x` | float | Accelerometer X-axis |
| `accel_y` | float | Accelerometer Y-axis |
| `accel_z` | float | Accelerometer Z-axis |
| `sleep_stage` | string | Sleep stage (W, N1, N2, N3, REM) from Dreem 2 EEG |
| `sleep_stage_onset` | datetime | Onset time of current sleep stage |

---

## 9. ScientISST MOVE

- **Status:** NO — requires PhysioNet credentials + accepted terms of use
- **Access:** `wfdb.dl_database("scientisst-move-biosignals/1.0.1", ...)`
- **Credentials:** `PHYSIONET_USERNAME` + `PHYSIONET_PASSWORD` (or `WFDB_USERNAME` + `WFDB_PASSWORD`)
- **File format:** CSV/TSV
- **License:** Open access

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Participant ID |
| `timestamp` | datetime | Measurement timestamp |
| `accel_x` | float | Accelerometer X-axis |
| `accel_y` | float | Accelerometer Y-axis |
| `accel_z` | float | Accelerometer Z-axis |
| `gyro_x` | float | Gyroscope X-axis |
| `gyro_y` | float | Gyroscope Y-axis |
| `gyro_z` | float | Gyroscope Z-axis |
| `eda` | float | Electrodermal activity |
| `heart_rate` | float | Heart rate |
| `skin_temperature` | float | Skin temperature |
| `activity_label` | string | Activity label (naturalistic) |

---

## 10. DREAMT

- **Status:** NO — requires PhysioNet credentials + accepted terms of use
- **Access:** `wfdb.dl_database("dreamt/2.2.0", ...)`
- **Credentials:** `PHYSIONET_USERNAME` + `PHYSIONET_PASSWORD` (or `WFDB_USERNAME` + `WFDB_PASSWORD`)
- **File format:** CSV/TSV
- **License:** Open access

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Participant ID |
| `night` | integer | Recording night |
| `timestamp` | datetime | Measurement timestamp |
| `accel_x` | float | Accelerometer X-axis (Empatica E4) |
| `accel_y` | float | Accelerometer Y-axis |
| `accel_z` | float | Accelerometer Z-axis |
| `heart_rate` | float | Heart rate |
| `eda` | float | Electrodermal activity |
| `skin_temperature` | float | Skin temperature |
| `bvp` | float | Blood volume pulse |
| `sleep_stage` | string | Sleep stage (W, N1, N2, N3, REM) from PSG |

---

## 11. MHEALTH

- **Status:** NO — UCI ML Repo dataset exists but `ucimlrepo.fetch_ucirepo(id=319)` fails (API issue as of early 2026)
- **Access:** `ucimlrepo.fetch_ucirepo(id=319)`
- **Credentials:** None
- **File format:** CSV (via UCI API)
- **License:** CC BY 4.0

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Subject ID (inferred from data structure) |
| `timestamp` | datetime | Measurement timestamp |
| `chest_accel_x` | float | Chest accelerometer X-axis |
| `chest_accel_y` | float | Chest accelerometer Y-axis |
| `chest_accel_z` | float | Chest accelerometer Z-axis |
| `chest_gyro_x` | float | Chest gyroscope X-axis |
| `chest_gyro_y` | float | Chest gyroscope Y-axis |
| `chest_gyro_z` | float | Chest gyroscope Z-axis |
| `chest_mag_x` | float | Chest magnetometer X-axis |
| `chest_mag_y` | float | Chest magnetometer Y-axis |
| `chest_mag_z` | float | Chest magnetometer Z-axis |
| `chest_ecg` | float | Chest ECG lead |
| `wrist_accel_x` | float | Wrist accelerometer X-axis |
| `wrist_accel_y` | float | Wrist accelerometer Y-axis |
| `wrist_accel_z` | float | Wrist accelerometer Z-axis |
| `ankle_accel_x` | float | Ankle accelerometer X-axis |
| `ankle_accel_y` | float | Ankle accelerometer Y-axis |
| `ankle_accel_z` | float | Ankle accelerometer Z-axis |
| `activity` | string | Activity label (12 activities) |

---

## 12. ExtraSensory

- **Status:** NO — server `extrasensory.ucsd.edu` unreachable (down since mid-2025)
- **Access:** Direct download from `https://extrasensory.ucsd.edu/extra_sensory_data/`
- **Credentials:** None
- **File format:** CSV/ZIP
- **License:** Public domain

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | User UUID |
| `timestamp` | datetime | Measurement timestamp |
| `activity` | string | Primary activity label |
| `accelerometer_x` | float | Phone accelerometer X-axis |
| `accelerometer_y` | float | Phone accelerometer Y-axis |
| `accelerometer_z` | float | Phone accelerometer Z-axis |
| `gyroscope_x` | float | Phone gyroscope X-axis |
| `gyroscope_y` | float | Phone gyroscope Y-axis |
| `gyroscope_z` | float | Phone gyroscope Z-axis |
| `compass_x` | float | Magnetometer/compass X-axis |
| `compass_y` | float | Magnetometer/compass Y-axis |
| `compass_z` | float | Magnetometer/compass Z-axis |
| `audio_level` | float | Microphone audio level |
| `location_lat` | float | GPS latitude |
| `location_lon` | float | GPS longitude |
| `mood` | string | Self-reported mood |
| `social_context` | string | Social context (alone, with friends, etc.) |
| `phone_usage` | string | Phone usage state |

---

## 13. All of Us Fitbit

- **Status:** NO — cloud-only access via Researcher Workbench, requires institutional DURA + 4-8 week application
- **Access:** BigQuery tables via `https://workbench.researchallofus.org`
- **Credentials:** AoU Researcher Workbench account (CITI training + DURA + research application)
- **File format:** BigQuery tables (no local download allowed)
- **License:** Controlled tier

### Tables

**activity_summary** (~39M rows)

| Column | Type | Description |
|--------|------|-------------|
| `person_id` | string | AoU participant ID |
| `date` | date | Activity date |
| `steps` | integer | Daily step total |
| `calories` | float | Calories burned |
| `distance_meters` | float | Distance in meters |
| `floors_climbed` | integer | Floors climbed |

**steps_intraday**

| Column | Type | Description |
|--------|------|-------------|
| `person_id` | string | Participant ID |
| `datetime` | timestamp | Minute-level timestamp |
| `steps` | integer | Steps in that minute |

**sleep_daily_summary** (~31M rows)

| Column | Type | Description |
|--------|------|-------------|
| `person_id` | string | Participant ID |
| `sleep_date` | date | Sleep date |
| `is_main_sleep` | boolean | Main sleep session? |
| `minute_asleep` | integer | Minutes asleep |

**sleep_level**

| Column | Type | Description |
|--------|------|-------------|
| `person_id` | string | Participant ID |
| `sleep_date` | date | Sleep date |
| `sleep_level` | string | Sleep level (awake, restless, resting, etc.) |

**heart_rate_summary**

| Column | Type | Description |
|--------|------|-------------|
| `person_id` | string | Participant ID |
| `date` | date | Date |
| `resting_heart_rate` | float | Resting HR |
| `heart_rate_minutes` | float | Minutes with HR data |

**heart_rate_minute_level**

| Column | Type | Description |
|--------|------|-------------|
| `person_id` | string | Participant ID |
| `datetime` | timestamp | Minute-level timestamp |
| `heart_rate` | float | Heart rate |

**person**

| Column | Type | Description |
|--------|------|-------------|
| `person_id` | string | Participant ID |
| `birth_datetime` | timestamp | Date of birth |

**device**

| Column | Type | Description |
|--------|------|-------------|
| `person_id` | string | Participant ID |
| `device_type` | string | Fitbit device model |

---

## 14. StepCountJITAI

- **Status:** NO — restricted academic access (contact paper authors)
- **Access:** Not publicly hosted; described in arXiv:2411.00336
- **Credentials:** N/A
- **File format:** CSV (expected)
- **License:** Academic

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Participant ID |
| `timestamp` | datetime | Observation timestamp |
| `step_count` | float | Step count |
| `intervention_type` | string | JITAI intervention delivered |
| `intervention_context` | string | Context at time of intervention |
| `time_of_day` | integer | Hour of day (0–23) |
| `day_of_week` | integer | Day of week |
| `activity_level` | float | Aggregate activity level |

---

## Credential Setup

### Kaggle API

Required for: PMData, WESAD, Fitbit Tracker.

1. Create a Kaggle account at https://www.kaggle.com
2. Settings -> API -> Create New Token
3. Save `kaggle.json` to `~/.kaggle/kaggle.json`
4. `chmod 600 ~/.kaggle/kaggle.json`

Or set `KAGGLE_API_TOKEN` env var to the file contents.

### PhysioNet

Required for: BIDSleep, ScientISST MOVE, DREAMT.

1. Register at https://physionet.org/register/
2. For each dataset, accept the terms of use on its PhysioNet page
3. Set env vars:
   ```bash
   export PHYSIONET_USERNAME="your_username"
   export PHYSIONET_PASSWORD="your_password"
   ```
   (or `WFDB_USERNAME` / `WFDB_PASSWORD`)

The loader creates a temporary `~/.netrc` if credentials exist.

### All of Us Researcher Workbench

Required for: All of Us Fitbit (4-8 week process).

1. Register at https://researchallofus.org
2. Complete CITI human subjects training ("Data or Specimens Only")
3. Obtain DURA from your institution
4. Submit a research application
5. Access BigQuery tables in the cloud Workbench

Data cannot leave the Workbench.

---

## Running the EDA Script

```bash
# Default (caches data under ./data/)
python scripts/eda_all_datasets.py

# Custom data directory
python scripts/eda_all_datasets.py --data-dir /path/to/data
```

What happens:
1. Each loader runs sequentially via `load_all()`
2. Datasets needing missing credentials log a warning and skip
3. A terminal report prints per-dataset: shape, dtypes, missing %, numeric summaries, categorical counts, temporal range, user count, correlations
4. A copy saves to `{data_dir}/eda_report.txt`

Only datasets marked **YES** produce output. Failed/skipped datasets show "** Dataset not loaded **".
