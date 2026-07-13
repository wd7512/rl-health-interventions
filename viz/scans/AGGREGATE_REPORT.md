# Research Landscape Scan — Aggregate Report

**Date:** 2026-01-13  
**Scope:** 2020-2026  
**Total Papers Discovered:** 247 (242 unique after deduplication)  
**Validation Status:** 111 valid, 38 title mismatches, 67 API errors (rate-limited), 30 non-DOI/arXiv IDs

---

## Executive Summary

Ten parallel subagents scanned the research landscape across 10 domains relevant to config-driven RL for Just-in-Time Adaptive Interventions (JITAIs) in physical activity. The scan identified **242 unique papers** from 2020-2026, including **negative results and failed trials** as requested.

### Coverage by Mission

| Mission | Topic | Papers |
|---------|-------|--------|
| 01 | RL for Physical Activity (2024-2026) | 19 |
| 02 | MRT Design & Statistical Methods | 24 |
| 03 | Contextual Bandits in Healthcare | 24 |
| 04 | Thompson Sampling & Bayesian Methods | 23 |
| 05 | Offline RL for Health | 25 |
| 06 | Wearable Sensor Processing | 27 |
| 07 | Step Count Modeling & Biomechanics | 25 |
| 08 | JITAI Literature (Broader) | 25 |
| 09 | Behavioral Frameworks & Theory | 30 |
| 10 | Digital Health Platforms | 25 |

---

## Key Findings by Domain

### 1. RL for Physical Activity (Mission 01)

**Critical papers:**
- **Ghosh et al. 2024** "Did we personalize?" — Resampling test on HeartSteps V2 data suggests personalization may be artifact of algorithm stochasticity
- **Karine & Marlin 2025** "Enhancing Adaptive Behavioral Interventions with LLM Inference" — LLM-inferred states improve RL by 15-25%
- **Song et al. 2025** "cMABxLLM" — First deployed hybrid contextual bandit + LLM for PA

**Negative result:** Kurisu et al. 2026 — RL app for obesity showed no significant PA improvement

### 2. MRT Methods (Mission 02)

**Core:** WCLS estimator (Qian et al. 2020, 2021) underpins all MRT methodology.

**Expansions:** Doubly robust estimation, categorical treatments, missing data handling, power-constrained bandits (ROGUE-TS), hybrid SMART-MRT designs.

**Gap:** No systematic review of failed/underpowered MRTs.

### 3. Contextual Bandits Healthcare (Mission 03)

24 papers across mental health (6), clinical trials (5), medication adherence (4), diabetes (2), ethics (2).

**Negative:** Hu & Duan 2025 — UCB/TS fail under context misreporting. Gonzalez et al. 2026 — Real-world deployment challenges documented.

**Dominant algorithm:** Thompson Sampling (12/24 papers).

### 4. Thompson Sampling Advances (Mission 04)

23 papers. **Negative:** Wang 2021 — "disappointing results with Thompson samplers" in clinical trials.

**Deployments:** TB screening Peru (Zhang 2025), COVID-19 testing Ohio (Rayo 2023).

### 5. Offline RL Health (Mission 05)

25 papers. CQL (8), BCQ (4), IQL (2).

**Negative:** Luo et al. ICML 2024 — RL can underperform random baselines. IQL collapses to behavior cloning.

**Bottom line:** CQL with transition sampling most validated. IQL needs geometric regularization.

### 6. Wearable Sensors (Mission 06)

27 papers. **NormWear validated** as 768-dim PPG/accelerometer foundation model.

**Negative:** FairTune — fine-tuning FMs widens gender gaps. BCoughBench — FM degradation under realistic conditions.

### 7. Step Count Modeling (Mission 07)

25 papers. **Negative:** MAPE >30% for wheelchair users. All devices fail below 0.8 m/s. Consumer trackers poor at individual precision.

### 8. JITAI Broader (Mission 08)

25 papers across 10 domains. **8 negative/null results.** Engagement paradox: needing intervention most when least able to engage.

### 9. Behavioral Frameworks (Mission 09)

30 papers, 9 frameworks. COM-B useful but gaps in digital engagement. Cultural factors moderate all domains.

### 10. Digital Platforms (Mission 10)

25 papers. **Negative:** 60-96% re-identification from step+HR. 15-30% data corrupted. No platform provides raw sensor access.

---

## Top 20 Recommendations for citations.json

1. **Ghosh et al. 2024** "Did we personalize?" — Validates HeartSteps V2 reproduction
2. **Karine & Marlin 2025** "Enhancing Adaptive Interventions with LLM Inference" — Sensor FM + LLM
3. **Song et al. 2025** "cMABxLLM" — First deployed hybrid CB + LLM
4. **NormWear (2024/2025)** — 768-dim PPG/accelerometer foundation model
5. **Li et al. 2025** "Power-constrained nonstationary bandits (ROGUE-TS)"
6. **Liu et al. 2025** "TS for zero-inflated count outcomes" (Drink Less study)
7. **Zhang et al. 2025** "Replicable bandits for post-deployment inference"
8. **Huch et al. 2024** "Mixed-effects Thompson Sampling"
9. **Qi et al. 2025** "Non-stationary TS with change-point detection"
10. **CQL with transition sampling (2024/2025)** — Most validated offline RL for health
11. **Geo-IQL (2024/2025)** — Prevents IQL collapse
12. **DynImp (2024/2025)** — Imputation at >50% missing
13. **FairTune (2024/2025)** — Bias amplification in FMs
14. **DR-WCLS (Shi & Dempsey 2023)** — Doubly robust MRT estimation
15. **Hybrid SMART-MRT (Li et al. 2026)**
16. **Kurisu et al. 2026** — Negative: RL app no effect
17. **Hu & Duan 2025** — Negative: UCB/TS fail under misreporting
18. **Nahum-Shani et al. 2026** — Engagement paradox
19. **Chikwetu et al. 2023** — 60-96% re-identification risk
20. **BCoughBench (2024/2025)** — FM degradation under realistic conditions

---

## Validation Notes

- **111 papers** validated against CrossRef/arXiv APIs
- **38 papers** have valid DOIs but title mismatches (short labels vs full titles)
- **67 papers** hit arXiv rate limiting (429 errors)
- **30 papers** use PubMed IDs (not validated)
- **5 papers** appeared in multiple missions (deduplicated)

---

## Files Generated

```
viz/scans/
├── mission_01_rl_pa.json + .md
├── mission_02_mrt_methods.json + .md
├── mission_03_bandits_healthcare.json + .md
├── mission_04_thompson_sampling.json + .md
├── mission_05_offline_rl_health.json + .md
├── mission_06_wearable_sensors.json + .md
├── mission_07_step_count.json + .md
├── mission_08_jitai_broader.json + .md
├── mission_09_behavioral_frameworks.json + .md
├── mission_10_platforms.json + .md
├── validate_dois.py
├── normalize_missions.py
├── aggregate.py
├── validation_report.json
└── aggregated.json (242 unique papers)
```

---

## Next Steps

1. Review top 20 recommendations and add to `viz/citations.json`
2. Fetch full reference lists for new main papers
3. Run `viz/curate.py` to compute evidence scores
4. Run `viz/frontier.py` to update frontier propagation
5. Update visualization to include new papers
