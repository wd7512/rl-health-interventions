# Fitbit Fitness Tracker Dataset

**Date:** 2026-06-10
**Context:** RL Health Interventions — Dataset Discovery

---

## Overview

| Property | Value |
|----------|-------|
| Source | Kaggle |
| URL | https://www.kaggle.com/datasets/haseeb85/fitbit-fitness-tracker-data |
| License | CC0 (Public Domain) |
| Participants | 33 users |
| Format | CSV (18 files, daily/hourly/minute-level aggregates) |
| Relevance | HIGH — real Fitbit data matching MDP wearable signals |

## Data Fields

| Field | Granularity | MDP Variable |
|-------|-------------|--------------|
| Steps | Daily, hourly, minute | steps_t |
| Heart Rate | Minute-level | hr_t |
| Sleep (duration, stages) | Daily | sleep_hours_t |
| Calories | Daily, hourly | Feature for baseline_activity |
| Distance | Daily | Feature for baseline_activity |
| Intensity minutes | Daily | sedentary_min_t (derived) |

## Relevance to Our Framework

| MDP Component | How Fitbit Tracker Helps |
|---|---|
| State variables | Steps, HR, sleep, calories — direct MDP variable sources |
| Synthetic parameterization | Real Fitbit distributions for generator seeding |
| Daily aggregation | Tests multi-timescale feature pipeline |
| Wearable form factor | Wrist-worn device matches real deployment scenario |

## Access

- Open access on Kaggle
- CC0 license (public domain)
- Downloadable immediately

## Files Reference

| File | Purpose |
|------|---------|
| `config/datasets/fitbit_tracker.yaml` | Data ingestion config |
