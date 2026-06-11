# 4TU Physical Activity Coaching Datasets

**Date:** 2026-06-10
**Context:** RL Health Interventions — Dataset Discovery

---

## Overview

| Property | Value |
|----------|-------|
| Source | 4TU.ResearchData (https://data.4tu.nl/) |
| License | CC BY 4.0 |
| Type | RL intervention datasets for PA coaching |
| Relevance | **HIGH** — only open-access datasets with intervention + step response |

## Datasets

### 1. Collaboratively Setting Daily Step Goals with a Virtual Coach
- **URL:** https://data.4tu.nl/datasets/53f2d238-77fc-4045-89a9-fb7fa2871f1d/1
- **Data:** Step count data from intervention study with virtual coach
- **MDP variables:** steps_t, goal_progress_t, intervention response

### 2. Dyadic Physical Activity Planning with a Virtual Coach
- **URL:** https://data.4tu.nl/datasets/2796f502-0610-4a7d-a8ee-ebc36639e0b1/1
- **Data:** PA planning data with coach interaction
- **MDP variables:** steps_t, previous_action, previous_response

### 3. Reinforcement Learning to Personalize Daily Step Goals
- **URL:** https://data.4tu.nl/datasets/6f8e6750-7494-4226-b6f9-299a9edbb077/1
- **Data:** Step count data with RL-based goal personalization
- **MDP variables:** steps_t, goal_progress_t, adaptive interventions

### 4. Feasibility of Motivational Messages for PA Coaching
- **URL:** https://data.4tu.nl/datasets/33888406-2d4e-4365-bf6e-0a45616842ef/1
- **Data:** PA data + motivational message response data
- **MDP variables:** steps_t, intervention response, message engagement

### 5. RL for Smoking Cessation/Physical Activity Interventions
- **URL:** https://data.4tu.nl/datasets/c7bf49ae-c6cf-4508-983d-c1ef37240d7f/3
- **Data:** Behavioral intervention data combining PA and smoking cessation
- **MDP variables:** steps_t, behavioral response, intervention delivery

## Relevance to Our Framework

| MDP Component | How 4TU Datasets Help |
|---|---|
| Action → Response | Open intervention response data for 1C calibration |
| RL benchmarks | RL-personalized step goals for agent comparison |
| Fatigue/burden | Engagement decay over intervention periods |
| Goal progress | Real goal-setting dynamics |

## Access

- All datasets CC BY 4.0
- Downloadable from 4TU.ResearchData
- No application required

## Files Reference

| File | Purpose |
|------|---------|
| `design.tex` | Formal MDP definition |
| `config/datasets/4tu_step_goals.yaml` | Data ingestion config (primary dataset) |
