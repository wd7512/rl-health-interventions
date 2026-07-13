# Mission 03: Contextual Bandits in Healthcare (2020-2026)

**Mission ID:** 03 | **Topic:** Contextual Bandits in Healthcare | **Date:** 2026-07-13 | **Papers:** 24

---

## Executive Summary

This scan covers contextual bandit (CB) applications in healthcare beyond physical activity. The field is moving from methodology to real-world deployment. Key findings: **mental health** dominates (6 papers incl. JAMA RCT, N=1,282), **medication adherence** has landmark trials (REINFORCE/npj Digital Medicine), **clinical trial design** is growing (dose-finding, adaptive randomization), and **ethical concerns** are being addressed via equity-constrained bandits. Negative results highlight strategic context misreporting, delayed RL benefits, and practical deployment difficulties.

---

## Domain Breakdown

### Mental Health (6 papers)
| Paper | Year | Result | Key Finding |
|-------|------|--------|-------------|
| Newby et al., JAMA Network Open | 2025 | Positive | CB allocated 1,282 students to digital interventions; PA & mindfulness best for severe distress |
| Kumar et al., AAAI (IAAI-24) | 2024 | Positive | TS bandit deployed with 1,100 mental health users via text messaging |
| Ameko et al., RecSys 2020 | 2020 | Positive | First offline CB for emotion regulation (n=114) |
| Yu et al., MOBILESoft 2024 | 2024 | Neutral | CAREForMe framework: CB with mobile sensing for mental health |
| Wang & Yang, IH 2026 | 2026 | **Mixed** | CB for journaling (N=38); benefits appeared only post-intervention |
| Huckvale et al., BMJ Open | 2023 | Neutral | Vibe Up protocol for CB-based adaptive trial |

### Medication Adherence (4 papers)
| Paper | Year | Result | Key Finding |
|-------|------|--------|-------------|
| Lauffenburger et al., npj Digital Medicine | 2024 | Positive | REINFORCE trial: RL improved adherence significantly at 6 months |
| Mate et al., NeurIPS 2020 | 2020 | Positive | Collapsing bandits for TB medication adherence |
| Biswas et al., arXiv 2021 | 2021 | Positive | Adaptive RMAB for diabetes/hypertension adherence |
| Pulick et al., arXiv 2026 | 2026 | Neutral | UCB-BOLD for digital therapeutics with endogenous adherence |

### Clinical Trial Design (5 papers)
| Paper | Year | Result | Key Finding |
|-------|------|--------|-------------|
| Varatharajah & Berry, Life 2022 | 2022 | Positive | CB + TS: 72.63% gain over random on stroke trial data |
| Wang & Tiwari, J Biopharm Stat 2026 | 2026 | Positive | CB for oncology dose-finding (BMS case study) |
| Shrestha & Jain, Statistics in Medicine 2021 | 2021 | Positive | TS bandit for N-of-1 clinical trials |
| Kanrar et al., arXiv 2025 | 2025 | Neutral | Risk-inclusive CB for early phase trials (Pfizer) |
| Wang, Pharmaceutical Statistics 2021 | 2021 | Neutral | Accelerated TS for response-adaptive designs |

### Diabetes (2 papers)
| Paper | Year | Result | Key Finding |
|-------|------|--------|-------------|
| Hotta et al., JMIR Form Res 2026 | 2026 | Positive | MAB for postprandial glucose; 23% improvement in pilot (n=6) |
| Killian et al., JMIR Diabetes 2024 | 2024 | Positive | RMAB with equity: +10% glucose targets, -85% disparities |

### Smoking Cessation (1 paper)
| Paper | Year | Result | Key Finding |
|-------|------|--------|-------------|
| Albers et al., PLOS ONE 2022 | 2022 | Positive | RL modeling current/future states for smoking cessation + PA |

### Hypertension (1 paper)
| Paper | Year | Result | Key Finding |
|-------|------|--------|-------------|
| Gonzalez et al., Contemp Clin Trials 2026 | 2026 | Neutral | LOWSALT4LIFE 2: CB deployment challenges documented |

### Ethics, Fairness & Negative Results (3 papers)
| Paper | Year | Result | Key Finding |
|-------|------|--------|-------------|
| Hu & Duan, AAMAS 2025 | 2025 | Neutral | **UCB and TS fail** under context misreporting in clinical trials |
| Chien et al., ACM FAccT 2022 | 2022 | Neutral | Multidisciplinary fairness analysis for bandits in clinical trials |
| Herlihy et al., KDD 2023 | 2023 | Neutral | Probabilistic fairness framework for RMAB health allocation |

### Other (3 papers)
| Paper | Year | Venue | Domain |
|-------|------|-------|--------|
| Zhou et al., 2023 | 2023 | Info Sys Research | Healthcare decision recommendations with diversity |
| Perianez et al., 2024 | 2024 | arXiv | HIV patient engagement in resource-limited settings |
| Galozy et al., 2023 | 2023 | arXiv | Bandit with state evolution + corrupted context (mHealth) |

---

## Algorithmic Trends

- **Thompson Sampling** dominates: 12/24 papers use TS or Bayesian approaches
- **Restless MABs** growing for population-level health allocation
- **Offline evaluation** (doubly robust, off-policy) critical for pre-deployment validation
- **Fairness constraints** increasingly integrated into bandit optimization

## Clinical Trial Maturity

| Stage | Count | Examples |
|-------|-------|----------|
| Simulation only | 12 | Dose-finding, offline evaluation methods |
| Protocol/Framework | 3 | Vibe Up protocol, CAREForMe |
| Pilot deployment | 5 | Glucose MAB (n=6), mental health transitions (n=38) |
| Mid-scale | 2 | Mental health (n=1,100), emotion regulation (n=114) |
| Large-scale RCT | 2 | Vibe Up (n=1,282), REINFORCE |

---

## Key Gaps

1. **Diabetes** underexplored despite high potential for personalized glucose management
2. **Smoking cessation** has only 1 relevant paper -- significant gap
3. **Failed deployments** underreported; only 2 papers document negative findings
4. **Long-term effects** rarely studied (only Wang & Yang 2026 examines post-intervention)
5. **Clinician-in-the-loop** bandits not yet explored in literature

## Connection to HeartSteps Context

- **IntelligentPooling** (Tomkins et al., 2021): Directly from Murphy lab; theoretical foundation for TS in mHealth
- **Power Constrained Bandits** (Yao et al., 2021): Addresses personalization vs. statistical power tension
- **LOWSALT4LIFE 2** (Gonzalez et al., 2026): Practical deployment challenges for HeartSteps-style trials
- **Equitable RMAB** (Killian et al., 2024): Fairness-aware allocation for diverse populations
- Diabetes, mental health, and medication adherence domains represent expansion opportunities

---

## Paper Quick Reference (Index by ID)

### DOIs
- `10.1007/s10994-021-05995-8` -- IntelligentPooling: Practical TS for mHealth (Tomkins 2021)
- `10.3390/life12081277` -- CB for Clinical Trial Decision-Making (Varatharajah 2022)
- `10.1001/jamanetworkopen.2025.40502` -- Vibe Up: AI-Enhanced Adaptive RCT (Newby 2025)
- `10.1136/bmjopen-2022-066249` -- Vibe Up Protocol (Huckvale 2023)
- `10.1609/aaai.v38i21.30328` -- Adaptive Bandit Experiments Mental Health (Kumar 2024)
- `10.1145/3383313.3412244` -- Offline CB Emotion Regulation (Ameko 2020)
- `10.2196/70826` -- Glucose Management MAB (Hotta 2026)
- `10.2196/52688` -- Equitable RMAB T2D (Killian 2024)
- `10.1016/j.cct.2026.108354` -- LOWSALT4LIFE 2 Online Learning (Gonzalez 2026)
- `10.1038/s41746-024-01028-5` -- REINFORCE Trial (Lauffenburger 2024)
- `10.1371/journal.pone.0277295` -- RL Smoking Cessation (Albers 2022)
- `10.1002/sim.8873` -- N-of-1 Bayesian Bandit (Shrestha 2021)
- `10.1145/3531146.3533154` -- Fairness ML Clinical Trials (Chien 2022)
- `10.1145/3580305.3599467` -- Probabilistic Fairness RMAB (Herlihy 2023)
- `10.1080/10543406.2025.2469877` -- Dose Selection CB (Wang & Tiwari 2026)
- `10.1287/isre.2022.1191` -- Healthcare Decision MAB (Zhou 2023)
- `10.1002/pst.2098` -- Accelerated TS Adaptive Trials (Wang 2021)

### arXiv IDs
- `arxiv:2401.15188` -- CAREForMe Mental Health Framework (Yu 2024)
- `arxiv:2606.06800` -- RL Clinical-Wellness Transitions (Wang & Yang 2026)
- `arxiv:2011.07989` -- Bandit State Evolution + Corrupted Context (Galozy 2023)
- `arxiv:2105.07965` -- Learn to Intervene RMAB (Biswas 2021)
- `arxiv:2408.07629` -- HIV Patient Engagement (Perianez 2024)
- `arxiv:2507.22344` -- Risk-inclusive CB Early Trials (Kanrar 2025)
- `arxiv:2605.24261` -- Digital Therapeutics Endogenous Adherence (Pulick 2026)
- `arxiv:2501.03865` -- Truthful Mechanisms Bandit Games (Hu & Duan 2025)
