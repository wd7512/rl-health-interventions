---
title: "DREAMT — Dataset for Real-time Sleep Stage Estimation Using Multisensor Wearable Technology"
status: "draft"
last_reviewed: "2026-07-12"
---

# DREAMT — Dataset for Real-time Sleep Stage Estimation Using Multisensor Wearable Technology

**Date:** 2026-06-10
**Context:** RL Health Interventions — Dataset Discovery

---

## Overview

| Property | Value |
|----------|-------|
| Source | PhysioNet |
| URL | https://physionet.org/content/dreamt/2.2.0/ |
| License | Open Access |
| Participants | 100 (Duke University Health System) |
| Sensors | Empatica E4 wristband + polysomnography (gold standard) |
| Relevance | HIGH — largest open wearable+sleep dataset |

## Data Fields

| Field | Source | MDP Variable |
|-------|--------|--------------|
| Accelerometer (3-axis) | Empatica E4 | Activity features |
| Heart rate (PPG) | Empatica E4 | hr_t |
| Electrodermal activity (EDA) | Empatica E4 | Autonomic features |
| Skin temperature | Empatica E4 | Feature |
| Blood volume pulse (BVP) | Empatica E4 | HR derivation |
| PSG sleep stage labels | Clinical | sleep_hours_t (derived) |

## Relevance to Our Framework

| MDP Component | How DREAMT Helps |
|---|---|
| Sleep modelling | 100 participants with gold-standard PSG labels |
| HR validation | Empatica E4 PPG validated against PSG |
| Autonomic features | EDA + temperature for user state estimation |
| Clinical population | Health intervention context |
| Synthetic validation | Validates wearable data distributions |

## Access

- Open access on PhysioNet
- Requires credentialing

## Files Reference

| File | Purpose |
|------|---------|
| `config/datasets/dreamt.yaml` | Data ingestion config |
