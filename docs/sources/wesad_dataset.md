---
title: "WESAD — Wearable Stress and Affect Detection"
status: "reference"
last_reviewed: "2026-07-12"
---

# WESAD — Wearable Stress and Affect Detection

**Date:** 2026-06-10
**Context:** RL Health Interventions — Dataset Discovery

---

## Overview

| Property | Value |
|----------|-------|
| Source | Kaggle / UCI |
| URL | https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset |
| License | Academic use |
| Participants | 15 subjects |
| Sensors | Chest (ECG, EDA, EMG, respiration, temperature) + wrist (accel, EDA, temp, BVP) |
| Labels | Stress, amusement, meditation, baseline |
| Relevance | HIGH — stress/affect labels for burden modelling |

## Sensor Layout

| Body Location | Sensors |
|---------------|---------|
| Chest | ECG, EDA, EMG, respiration, temperature |
| Wrist | Accelerometer, EDA, temperature, blood volume pulse (BVP) |

## Relevance to Our Framework

| MDP Component | How WESAD Helps |
|---|---|
| Burden_t | Stress labels map to intervention burden state |
| User response | Affect labels calibrate response model archetypes |
| Fatigue dynamics | Stress/amusement/meditation transitions over time |
| Wrist BVP | Same sensor type as consumer wearables (Fitbit PPG) |

## Access

- Available on Kaggle
- Academic use license

## Files Reference

| File | Purpose |
|------|---------|
| `config/datasets/wesad.yaml` | Data ingestion config |
