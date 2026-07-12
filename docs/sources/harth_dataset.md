---
title: "HARTH Dataset Family (Human Activity Recognition for Thigh and Back)"
status: "reference"
last_reviewed: "2026-07-12"
---

# HARTH Dataset Family (Human Activity Recognition for Thigh and Back)

**Date:** 2026-06-10
**Context:** RL Health Interventions — Dataset Discovery

---

## Overview

| Property | Value |
|----------|-------|
| Source | HuggingFace (josefheidler) |
| License | MIT |
| Type | Labelled activity recognition with body-worn accelerometers |
| Relevance | HIGH — MIT license, multi-demographic, fine-grained |

## Variants

### 1. HARTH Adults 2021
- **URL:** https://huggingface.co/datasets/josefheidler/har_adults_2021-harth
- **Participants:** 31 healthy adults
- **Sensors:** Dual Axivity AX3 accelerometers (thigh + lower back, 50 Hz)
- **Activities:** 12 types (walking, running, stair climbing, cycling, sitting, standing, etc.)
- **License:** MIT

### 2. HARTH Children 2024
- **URL:** https://huggingface.co/datasets/josefheidler/har_children_2024-harth
- **Participants:** Children
- **Sensors:** Same as adults
- **License:** MIT

### 3. HARTH Older Adults 2023
- **URL:** https://huggingface.co/datasets/josefheidler/har_older-adults_2023-harth
- **Participants:** Older adults
- **Sensors:** Same as adults
- **License:** MIT

### 4. HARTH Wrist Adults 2025
- **URL:** https://huggingface.co/datasets/josefheidler/har_ws_adults_2025-harth
- **Participants:** Adults, wrist-only variant
- **License:** MIT

## Relevance to Our Framework

| MDP Component | How HARTH Helps |
|---|---|
| Activity distributions | 12 activity types for synthetic data parameterization |
| User archetypes | Multiple demographics (adults, children, older adults) |
| Sensor validation | 50 Hz body-worn data validates phone-based approximations |
| Sedentary detection | Sitting/standing/lounging labels for sedentary_min_t |

## Access

- All variants on HuggingFace
- MIT license (most permissive)
- Directly loadable with HuggingFace `datasets` library

## Files Reference

| File | Purpose |
|------|---------|
| `config/datasets/harth.yaml` | Data ingestion config |
