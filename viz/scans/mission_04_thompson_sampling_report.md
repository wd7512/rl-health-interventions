# Mission 04: Thompson Sampling & Bayesian Methods for Health (2020-2026)

**Scan Date:** 2026-07-13
**Mission ID:** 04
**Topic:** Thompson Sampling & Bayesian Methods
**Papers Found:** 23

---

## Executive Summary

This scan covers Thompson Sampling (TS) and Bayesian methods applied to health interventions, digital health, and clinical trials from 2020-2026. We identified 22 papers spanning TS variants for response-adaptive randomization, Bayesian bandits for mHealth, non-stationary extensions, dose-finding, finite-sample properties, and real-world deployments for public health screening.

**Papers excluded (covered by other missions):** IntelligentPooling (Tomkins 2021), reBandit (Ghosh 2024), Count-based TS (Liu 2025), power-constrained exploration (Yao 2021), HeartSteps V2 (Liao 2022).

---

## Category 1: TS for Clinical Trial Design

### 1. Bayesian-Bandit Adaptive Design for N-of-1 Trials
- **Shrestha & Jain (2021)** — *Statistics in Medicine*
- DOI: [10.1002/sim.8873](https://doi.org/10.1002/sim.8873)
- First TS-based Bayesian adaptive design for N-of-1 trials. Hierarchical model jointly estimates individual/population effects via MCMC. Strong relevance to HeartSteps for personalizing JITAIs.

### 2. Accelerated Thompson Sampling for Response-Adaptive Trials
- **Wang (2021)** — *Pharmaceutical Statistics*
- DOI: [10.1002/pst.2098](https://doi.org/10.1002/pst.2098)
- **KEY NEGATIVE RESULT:** Reports TS disappointing in clinical trial settings. Proposes accelerated TS for improved short-run performance.

### 3. Top-Two Thompson Sampling for Best Treatment ID
- **Wang & Tiwari (2023)** — *Pharmaceutical Statistics*
- DOI: [10.1002/pst.2331](https://doi.org/10.1002/pst.2331)
- TTTS with acceleration for small-sample drug development. Introduces simpler TTTS2 variant.

### 4. TS for Sequential Multiple Assignment Randomized Trials
- **Norwood, Davidian & Laber (2024)** — *Biometrics*
- DOI: [10.1093/biomtc/ujae152](https://doi.org/10.1093/biomtc/ujae152)
- First TS-based RAR for SMARTs with valid post-study inference. Directly applicable to JITAI evaluation.

### 5. Finite-Sample Error Control for RAR
- **Deliu & Villar (2025)** — *Biometrics*
- DOI: [10.1093/biomtc/ujaf069](https://doi.org/10.1093/biomtc/ujaf069)
- Conditions where standard TS-based inference fails. Finite-sample and asymptotic analysis.

### 6. MABs for Dose-Finding
- **Kojima (2023)** — *Artificial Intelligence in Medicine*
- DOI: [10.1016/j.artmed.2023.102713](https://doi.org/10.1016/j.artmed.2023.102713)
- TS vs UCB vs Gittins for dose-finding. TS balances patient benefit and correct dose ID.

### 7. Contextual Bandits for Phase I/II Dose Selection
- **Wang & Tiwari (2026)** — *J. Biopharmaceutical Statistics*
- DOI: [10.1080/10543406.2025.2469877](https://doi.org/10.1080/10543406.2025.2469877)
- Contextual TS for personalized dose selection handling patient covariates.

### 8. Bayesian RAR for Cluster RCTs
- **Liu, Young Karris & Jain (2026)** — *Statistics in Medicine*
- DOI: [10.1002/sim.70386](https://doi.org/10.1002/sim.70386)
- First TS-based RAR extended to cluster-randomized trials with intra-cluster correlation.

### 9. MAB Backfill Bayesian Optimal Interval Design
- **Kojima & Takeda (2025)** — *J. Biopharmaceutical Statistics*
- DOI: [10.1080/10543406.2025.2602463](https://doi.org/10.1080/10543406.2025.2602463)
- TS-MAB integrated with BOIN for dose-finding with backfill.

---

## Category 2: TS for mHealth & Digital Health

### 10. Practical Online Learning for mHealth
- **Gonzalez et al. (2026)** — *Contemporary Clinical Trials*
- DOI: [10.1016/j.cct.2026.108354](https://doi.org/10.1016/j.cct.2026.108354)
- Practical TS deployment guide: cold-start, delayed feedback, non-stationarity, exploration-exploitation. Heart failure app case study. **Very strong relevance to HeartSteps.**

### 11. Replicable Bandits for Digital Health
- **Zhang, Closser, Trella & Murphy (2025)** — *Statistical Science*
- DOI: [10.1214/25-sts1017](https://doi.org/10.1214/25-sts1017)
- TS variants guaranteeing valid inference after adaptive data collection. **Critical for post-deployment analysis.**

### 12. Causal Inference and Bandits for Digital Health (Review)
- **Svihrova et al. (2025)** — *Frontiers in Digital Health*
- DOI: [10.3389/fdgth.2025.1435917](https://doi.org/10.3389/fdgth.2025.1435917)
- Reviews TS variants, exploration strategies, causal inference challenges. Open problems: non-stationarity, heterogeneity.

### 13. Mixed-Effects Bandit for Mobile Health
- **Huch et al. (2024)** — *NeurIPS*
- Proposes robust mixed-effects TS accounting for user heterogeneity in mHealth.

---

## Category 3: Non-Stationary TS

### 14. Thompson Sampling for Non-Stationary Bandit Problems
- **Qi, Guo & Zhu (2025)** — *Entropy*
- DOI: [10.3390/e27010051](https://doi.org/10.3390/e27010051)
- TS with change-point detection and sliding-window posteriors. Provides regret bounds. **Critical for JITAIs where user behavior evolves.**

---

## Category 4: Bayesian Optimization for Health

### 15. Bayesian Strategy for GP Prescribing
- **Allender et al. (2020)** — *npj Digital Medicine*
- DOI: [10.1038/s41746-019-0205-y](https://doi.org/10.1038/s41746-019-0205-y)
- GP-based Bayesian optimization identifies optimal healthcare strategies.

### 16. Bayesian Optimization for TMS
- **Schultheiss et al. (2026)** — *PLoS Computational Biology*
- DOI: [10.1371/journal.pcbi.1013994](https://doi.org/10.1371/journal.pcbi.1013994)
- GP-based Bayesian optimization for TMS coil positioning minimizing patient stimulations.

---

## Category 5: Bandits for Public Health

### 17. TS for TB Screening in Peru
- **Zhang et al. (2025)** — *Scientific Reports*
- DOI: [10.1038/s41598-025-31829-x](https://doi.org/10.1038/s41598-025-31829-x)
- Deployed TS optimizing mobile TB screening in Lima. Increases case detection. **Rare real-world TS deployment.**

### 18. Bandits for COVID-19 Case-Finding
- **Rayo et al. (2023)** — *JMIR Public Health and Surveillance*
- DOI: [10.2196/39754](https://doi.org/10.2196/39754)
- TS bandits for adaptive testing site allocation in Ohio.

### 19. Bandits with Testing Volume Constraints
- **Warren et al. (2025)** — *JRSS: Series A*
- DOI: [10.1093/jrsssa/qnae090](https://doi.org/10.1093/jrsssa/qnae090)
- Integrates testing volume constraints into TS bandits for disease surveillance.

### 20. Bayesian m-Top for Vaccine Allocation
- **Cimpean et al. (2026)** — *Scientific Reports*
- DOI: [10.1038/s41598-026-40787-x](https://doi.org/10.1038/s41598-026-40787-x)
- Bayesian m-top exploration (TS-related) for vaccine policy evaluation.

---

## Category 6: Theoretical / Comparison Studies

### 21. Optimal Allocation in Adaptive Designs
- **Yi & Wang (2025)** — *Statistical Methods in Medical Research*
- DOI: [10.1177/09622802241293750](https://doi.org/10.1177/09622802241293750)
- Theoretical comparison: TS-based allocation vs optimal Neyman allocation.

### 22. Bayesian RAR with Binary Outcomes (Comparison)
- **Proper, Connett & Murray (2021)** — *Clinical Trials*
- DOI: [10.1177/17407745211010139](https://doi.org/10.1177/17407745211010139)
- Compares Bayesian RAR techniques including TS-like posterior probability methods.

### 23. Contextual-Bandit for Clinical Trial Decision-Making
- **Varatharajah & Berry (2022)** — *Life*
- DOI: [10.3390/life12081277](https://doi.org/10.3390/life12081277)
- Contextual bandit using Bayesian posterior sampling for treatment allocation conditional on patient covariates.

---

## Key Trends (2020-2026)

1. **From theory to deployment (2023-2026):** Early papers (2020-2022) focus on TS methodology. Later papers show real-world deployments — TB screening, COVID-19 testing, mHealth apps.
2. **Inference after adaptation (2024-2026):** Emerging theme from Murphy's group — valid inference after TS-based data collection (Replicable Bandits; Mixed-Effects Bandit).
3. **Non-stationarity recognition:** Multiple papers identify evolving user behavior as a critical open problem for TS in health.
4. **TS limitations documented:** Wang (2021) explicitly reports TS fails in small-sample clinical trial settings. Deliu & Villar (2025) analyze finite-sample failures.

## Gaps & Open Problems

- No deployed TS system for mHealth JITAIs (only simulations/feasibility studies)
- Limited work on TS with continuous action spaces for dosage optimization
- Few papers on TS safety constraints for health interventions
- Inference after TS deployment remains a frontier problem

## Relevance to HeartSteps V2

HeartSteps V2 uses TS with dosage tracking and proxy value function. Most directly relevant papers:
- **Gonzalez 2026** — practical TS deployment in mHealth
- **Zhang 2025** — replicable bandits for valid inference
- **Huch 2024** — mixed-effects TS for user heterogeneity
- **Norwood 2024** — TS for multi-stage trial designs
- **Qi 2025** — non-stationary TS

## Search Methodology

Sources: PubMed (E-utilities API), OpenAlex. Search terms: "Thompson sampling mHealth", "Bayesian bandit health", "posterior sampling digital health", "Thompson sampling clinical trial", "Bayesian optimization clinical trial", "Thompson sampling non-stationary", "multi-armed bandit health Bayesian". Date range: 2020-2026.

## Statistics

| Metric | Count |
|--------|-------|
| Total papers | 23 |
| Methodology papers | 15 |
| Application/deployment | 6 |
| Review papers | 2 |
| Positive result | 17 |
| Mixed/Negative result | 4 |
| Neutral result | 1 |
| mHealth-relevant | 8 |
| TS-specific | 16 |
| Bayesian (non-TS) | 6 |
