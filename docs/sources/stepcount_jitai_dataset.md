---
title: "StepCountJITAI Dataset"
status: "active"
last_reviewed: "2026-07-12"
---

# StepCountJITAI Dataset

**Date:** 2026-06-10
**Context:** RL Health Interventions — Dataset Discovery

---

## Overview

| Property | Value |
|----------|-------|
| Paper | Karine & Marlin (2024) |
| Title | "StepCountJITAI: simulation environment for RL with application to physical activity adaptive intervention" |
| Venue | NeurIPS 2024 Workshop on Behavioral ML |
| arXiv | https://arxiv.org/abs/2411.00336 |
| Type | RL simulation environment from real MRT data |
| Relevance | **HIGHEST** — purpose-built for RL-based PA JITAIs |

## What It Contains

- Step count trajectories simulated from real micro-randomized trial data
- Intervention action space (message types, timing, context)
- User response dynamics calibrated from real HeartSteps-style trials
- Contextual features: time of day, day of week, activity level

## Relevance to Our Framework

| MDP Component | How StepCountJITAI Helps |
|---|---|
| State space | Real context features validate our feature set |
| Action space | Message types map to our action set (a0-a5) |
| User response model | Observed responses — direct calibration for 1C |
| Agent benchmark | Purpose-built RL environment for PA intervention agents |
| Reward signals | Step count deltas ground our R_t weights |

## Access

- Available via the paper's supplementary materials
- Academic use
- GitHub repository linked in paper

## Files Reference

| File | Purpose |
|------|---------|
| `initial_design.tex` | Formal MDP definition |
| `config/datasets/stepcountjitai.yaml` | Data ingestion config |
