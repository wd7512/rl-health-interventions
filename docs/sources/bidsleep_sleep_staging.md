---
title: "BIDSleep Dataset — Multi-Night HR and Accelerometry with Sleep Labels"
status: "draft"
last_reviewed: "2026-07-12"
---

# BIDSleep Dataset — Multi-Night HR and Accelerometry with Sleep Labels

**Date:** 2026-06-10
**Context:** RL Health Interventions — Dataset Discovery

---

## Overview

| Property | Value |
|----------|-------|
| Source | PhysioNet |
| URL | https://physionet.org/content/bidsleep-dataset/1.0.0/ |
| License | Open Access |
| Sensors | Apple Watch (accelerometer + HR) + Dreem 2 headband (EEG) |
| Type | Multi-night wearable + clinical sleep staging |
| Relevance | HIGH — consumer wearable paired with gold-standard sleep labels |

## Data Fields

| Field | Source | MDP Variable |
|-------|--------|--------------|
| Instantaneous heart rate | Apple Watch | hr_t |
| 3-axis accelerometry | Apple Watch | Feature for activity |
| EEG sleep stage labels | Dreem 2 (gold standard) | sleep_hours_t (derived) |
| Multi-night recordings | Both | Night-to-night variability |

## Relevance to Our Framework

| MDP Component | How BIDSleep Helps |
|---|---|
| Sleep modelling | Gold-standard sleep stages validate sleep features |
| HR from wearable | Apple Watch PPG — consumer device, not clinical |
| Multi-night | Captures night-to-night sleep variability |
| Synthetic validation | Validates synthetic sleep data against real wearable recordings |

## Access

- Open access on PhysioNet
- Downloadable after credentialing

## Files Reference

| File | Purpose |
|------|---------|
| `config/datasets/bidsleep.yaml` | Data ingestion config |
