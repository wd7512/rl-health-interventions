# PMData — A Sports Logging Dataset

**Date:** 2026-06-10
**Context:** RL Health Interventions — Dataset Discovery

---

## Overview

| Property | Value |
|----------|-------|
| Source | Kaggle |
| URL | https://www.kaggle.com/datasets/vlbthambawita/pmdata-a-sports-logging-dataset |
| License | CC BY 4.0 |
| Size | 1 GB (912 files) |
| Sources | Garmin watch, WHOOP strap, HRV4Training, Apple Health |
| Relevance | HIGH — largest open fitness dataset, multi-source |

## Data Sources Within PMData

| Source | Data Available |
|--------|---------------|
| Garmin watch | HR, steps, HRV, sleep, SpO2, body temp |
| WHOOP strap | HR, HRV, sleep stages, recovery, strain |
| HRV4Training | HRV, resting HR, sleep quality |
| Apple Health | Steps, HR, calories, activity type |

## Relevance to Our Framework

| MDP Component | How PMData Helps |
|---|---|
| HR patterns | Continuous HR from multiple consumer devices |
| Sleep modelling | Sleep stages + quality from WHOOP |
| Recovery/burden | Training load and recovery scores for fatigue modelling |
| Activity distributions | Step counts from Garmin for synthetic parameterization |
| HRV features | Autonomic nervous system features for user state estimation |

## Access

- Open access on Kaggle
- CC BY 4.0 license
- Downloadable immediately (1 GB)

## Files Reference

| File | Purpose |
|------|---------|
| `config/datasets/pmdata.yaml` | Data ingestion config (Garmin sub-dataset) |
