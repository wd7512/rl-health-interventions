---
title: "MHEALTH (Mobile Health) Dataset"
status: "draft"
last_reviewed: "2026-07-12"
---

# MHEALTH (Mobile Health) Dataset

**Date:** 2026-06-10
**Context:** RL Health Interventions — Dataset Discovery

---

## Overview

| Property | Value |
|----------|-------|
| Source | UCI ML Repository |
| URL | https://archive.ics.uci.edu/dataset/319/mhealth+dataset |
| License | CC BY 4.0 |
| Participants | 10 subjects |
| Sensors | Chest (accel, gyro, mag, ECG) + wrist (accel, gyro, mag) + ankle (accel, gyro, mag) |
| Activities | 12 types (walking, running, cycling, sitting, standing, etc.) |
| Relevance | HIGH — multi-body sensor placement with ECG |

## Sensor Layout

| Body Location | Sensors | Sampling Rate |
|---------------|---------|---------------|
| Chest | Accelerometer, gyroscope, magnetometer, ECG | 50 Hz |
| Wrist | Accelerometer, gyroscope, magnetometer | 50 Hz |
| Ankle | Accelerometer, gyroscope, magnetometer | 50 Hz |

## Relevance to Our Framework

| MDP Component | How MHEALTH Helps |
|---|---|
| HR validation | Chest ECG validates wrist PPG estimates |
| Activity classification | 12 types for synthetic data seeding |
| Multi-sensor fusion | Body-worn sensors complement phone-based data |
| Sedentary detection | Sitting/standing labels for sedentary_min_t |

## Access

- Open access on UCI ML Repository
- CC BY 4.0 license
- Downloadable immediately

## Files Reference

| File | Purpose |
|------|---------|
| `config/datasets/mhealth.yaml` | Data ingestion config |
