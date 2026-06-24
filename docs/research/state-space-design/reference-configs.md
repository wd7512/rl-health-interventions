---
title: "Reference Configurations — State Space"
status: "draft v0.1"
date: "2026-06-23"
purpose: "Describe what 5+ published systems have used for state representation"
---

# Reference Configurations — State Space

> Descriptive, not prescriptive. Each entry reports what a published system
> actually implemented for its state representation.

## 1. HeartSteps V2 — Liao et al. (2019, ACM IMWUT)

| Component | Detail |
|-----------|--------|
| **State type** | Feature vector (no discrete state index) |
| **Features** | Time-of-day (5 bins: decision point), location (home/work/other), weather, recent activity (binary: active in last 30 min), day of week |
| **Cardinality** | Not enumerated — treated as continuous features for Thompson sampling |
| **Activity feature** | Binary: any steps in last 30 min vs none |
| **Step count** | Not used as a state feature |
| **Trend** | None |
| **Time features** | Decision point (5 levels), day of week (7 levels) |
| **Observability** | All features observable in real time (phone sensors, GPS, weather API) |

**Relevance**: Most successful deployed RL-for-health system. Feature-based
state (not tabular) with strong time-of-day effect.

## 2. HeartSteps V1 MRT — Klasnja et al. (2019, Annals of Behavioral Medicine)

| Component | Detail |
|-----------|--------|
| **State type** | Stratification variables for MRT (not RL features) |
| **Features** | Time-of-day (binary: morning/afternoon), recent activity (binary), weekend vs weekday |
| **Cardinality** | 2 × 2 × 2 = 8 strata |
| **Activity feature** | Binary: sedentary vs active in last 30 min |
| **Key result** | Sedentary state → suggestions 2× more effective |

**Relevance**: Established that a binary activity state is sufficient to
detect context-dependent treatment effects.

## 3. StepCountJITAI Simulation — Karine et al. (2024)

| Component | Detail |
|-----------|--------|
| **State type** | Discrete tabular states |
| **Features** | Activity level (2–4 bins depending on condition), time-of-day (4–5 bins), day type (weekday/weekend) |
| **Cardinality** | 2×5×2 = 20 (min) to 4×5×2 = 40 (max) |
| **Activity feature** | Multi-level bins (not binary) — closest to our proposal |
| **Time-of-day** | Mapped to decision points (0–4) |
| **Evaluation** | 4-bin outperforms 2-bin for policy learning |

**Relevance**: Direct empirical comparison showing 4-bin > 2-bin for state
representation in step-based RL.

## 4. Trella et al. (2022, Algorithms) — Design Guidelines

| Component | Detail |
|-----------|--------|
| **State type** | Framework — no specific state representation |
| **Key prescription** | State should include: (1) intervention history, (2) user context, (3) outcome history |
| **Cardinality guidance** | "Avoid combinatorial explosion by using function approximation rather than tabular encoding" |
| **Context features** | Recommends time-of-day, day-of-week, location, recent activity, stress (if available) |

**Relevance**: Design framework that validates our feature set. Recommends
factored over flat representation.

## 5. Health Gym — Gateno (2023) — Simulated environment

| Component | Detail |
|-----------|--------|
| **State type** | Continuous and discrete mixed |
| **Features** | Steps (count), sedentary minutes, active minutes, heart rate (if wearable), time of day, day of week |
| **Step representation** | Raw count (continuous) — not binned |
| **Cardinality** | Effectively infinite (continuous features) |
| **Evaluation** | Continuous representation requires function approximation; tabular infeasible |

**Relevance**: Counterpoint — Health Gym uses raw continuous steps rather
than bins. Comparison would test whether binning discards meaningful
variation.

## 6. All of Us / UK Biobank population studies

| Component | Detail |
|-----------|--------|
| **State type** | Observational — not an intervention system |
| **Features used** | Step count categories (population distribution), age, sex, BMI |
| **Step representation** | Quartile-based bins in most analyses |
| **Key finding** | Population step distributions are roughly normal (mean ~7k–8k) with long right tail |

**Relevance**: Validates that 4 bins cover the clinically meaningful range
without losing information in dense regions.

## Summary: State feature cardinality across systems

| System | Activity bins | Time-of-day | Day type | Trend | Hidden state |
|--------|-------------|-------------|----------|-------|-------------|
| HeartSteps V2 | Binary (active/not) | 5 (decision point) | Yes (weekday/weekend) | No | No |
| HeartSteps V1 MRT | Binary (sedentary/active) | 2 (AM/PM) | Yes | No | No |
| StepCountJITAI | 2–4 bins | 4–5 | Yes | No | No |
| Trella guidelines | — (framework) | Prescribed | Prescribed | — | Optional (stress) |
| Health Gym | Continuous (raw) | Yes | Yes | No | No |
| Proposed state space | 4 bins | 4 | Yes | Yes (3 levels) | Optional (stress/sleep) |

**Observation**: Trend is the only proposed dimension that no published
system has used. All others have precedent in at least one system.
Hidden mood/stress appears only in Trella as an optional recommendation,
never implemented.
