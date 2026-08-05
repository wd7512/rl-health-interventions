# Mission 05: Offline RL for Health (2020-2026)

**Mission ID:** `05`
**Topic:** Offline Reinforcement Learning for Health
**Date:** 2026-07-13
**Papers Found:** 25

---

## 1. Executive Summary

This scan covers offline/batch RL applied to healthcare (2020-2026). The field has evolved from algorithmic adaptations (CQL, BCQ, IQL) to safety-constrained frameworks, fairness-aware approaches, and large-scale audits. Dominant domains: sepsis, diabetes, ventilation, anesthesia. A critical sub-literature on **failure modes** provides cautionary evidence.

---

## 2. Methodology

**Sources:** arXiv API, Google Scholar, PubMed | **Queries:** 10 | **Papers screened:** ~300+ | **Selected:** 25

Queries: offline RL health, conservative Q-learning clinical, batch RL healthcare, IQL clinical, BCQ healthcare, offline policy learning medical, distribution shift offline RL health, safety constraints offline RL, failure modes offline RL.

---

## 3. Algorithm Distribution

| Algorithm | Count | Key Apps |
|-----------|-------|----------|
| CQL | 8 | Ventilation, anesthesia, diabetes, sepsis |
| BCQ | 4 | Transfusion, glucose, sepsis |
| IQL | 2 | Sepsis (w/ geometric pessimism) |
| Decision Transformer | 2 | Sepsis, mental health |
| SPIBB | 1 | Sepsis vasopressors |
| Conformal + Offline RL | 1 | Population health |
| Model-based Offline RL | 2 | MCS weaning, general medical |
| Other (AWR, CB, etc.) | 5 | Sepsis, warfarin, activity |

### Clinical Domains
Sepsis: 10 | Diabetes: 5 | Ventilation: 2 | Anesthesia: 1 | Transfusion: 1 | Warfarin: 2 | ICU Insulin: 1 | Population health: 1 | Physical activity: 1 | MCS weaning: 1

---

## 4. Negative Results and Failure Modes

### 4.1 Luo et al. (ICML 2024) -- "RL in DTRs Needs Critical Reexamination"
17,000+ experiments: RL can be **surpassed by random baselines** depending on OPE metric and MDP formulation. Calls into question many published positive results.

### 4.2 Nidadala & Guthi (2025) -- "Horizon Reduction as Information Loss"
Formal proof: horizon reduction makes optimal policies **statistically indistinguishable** from suboptimal ones. Three structural failure modes identified.

### 4.3 Wanjari (2026) -- IQL Collapses to BC
Standard IQL on MIMIC-III sepsis: 75% terminal agreement (= behavior cloning). Geo-IQL (geometric pessimism) recovers 86.4%.

### 4.4 Liu et al. (2021) -- Confounding Bias
Offline RL biased against aggressive treatment. Mitigation: subspace learning.

### 4.5 Pace et al. (2023) -- Hidden Confounding
Even nonidentifiable settings compromise validity. Mitigation: delphic uncertainty.

---

## 5. Core Papers (16)

1. **Tang & Wiens (2021)** -- Model Selection for Offline RL in Healthcare [MLHC]
2. **Liu et al. (2021)** -- Offline RL with Uncertainty for Sepsis
3. **Satija et al. (2021)** -- Multi-Objective SPIBB [safety guarantees]
4. **Fatemi et al. (2022)** -- Semi-Markov Offline RL for Healthcare [CHIL]
5. **Kondrup et al. (2022)** -- DeepVent: Safe Ventilation via CQL [IAAI]
6. **Emerson et al. (2022)** -- Offline RL for Blood Glucose Control [JBHI]
7. **Nambiar et al. (2023)** -- Deep Offline RL for Treatment Optimization [KDD]
8. **Pace et al. (2023)** -- Delphic Offline RL under Hidden Confounding
9. **Cai et al. (2023)** -- PCQL for Personalized Anesthesia [IEEE JBHI]
10. **Rahman et al. (2024)** -- Medical Decision Transformer for Sepsis
11. **Luo et al. (2024)** -- RL in DTRs Needs Critical Reexamination [ICML]
12. **Yan et al. (2025)** -- OGSRL: Offline Guarded Safe RL [theory + guarantees]
13. **Basu et al. (2025)** -- HACO: Conformal Offline RL [fairness auditing]
14. **Wanjari (2026)** -- Geo-IQL: Geometric Pessimism for Sepsis
15. **Yu et al. (2026)** -- T-CQL: Transformer CQL + Digital Twin
16. **Frost & Harris (2026)** -- Insulin4RL Dataset [new benchmark]

---

## 6. Supporting Papers (9)

17. Bennett & Kallus (2021) -- Proximal RL: OPE in POMDPs
18. Kuo et al. (2021) -- Health Gym synthetic datasets
19. Wang et al. (2022) -- BCQ + TL for blood transfusion
20. Kausik et al. (2022) -- OPE under confounding
21. Wang et al. (2022) -- Continuous-space sepsis treatment
22. Huang et al. (2024) -- Warfarin via offline contextual bandit
23. Kumar et al. (2026) -- Attention + LLM rationale for sepsis
24. Lin et al. (2026) -- Physical activity via All of Us
25. Tumay et al. (2025) -- CORMPO for MCS weaning

---

## 7. Key Datasets

| Dataset | Domain | Key Users |
|---------|--------|-----------|
| MIMIC-III | ICU/sepsis | Most papers |
| MIMIC-IV | ICU/insulin | Frost 2026, Ji 2024 |
| eICU | Multi-site | Zou 2025, Kumar 2026 |
| Health Gym (synthetic) | Sepsis/hypotension | Kuo 2021 |
| UVA/Padova | T1D | Emerson 2022, 2025 |
| All of Us | Activity | Lin 2026 |
| Insulin4RL | ICU insulin | Frost 2026 |

---

## 8. Gaps and Recommendations for Project

### Gaps
- IQL in healthcare is fragile (collapses without regularization)
- CQL dominant but has known limitations (action imbalance, random-baseline vulnerability)
- Model-based offline RL in healthcare is nascent (2 papers)
- OPE methodology contested; no evaluation standard
- Fairness under-explored (only 1 paper)
- Zero prospective clinical deployments found

### Recommendations
- **Primary algorithm:** CQL with transition sampling (Nambiar 2023)
- **If using IQL:** add geometric regularization (Wanjari 2026)
- **Use SMDP** (Fatemi 2022) for variable-time actions
- **OPE:** FQE (Tang 2021), cross-validate with multiple methods
- **Safety framework:** OGSRL (Yan 2025) or SPIBB (Satija 2021)
- **Synthetic-to-real calibration:** conformal approach (Basu 2025) or digital twin (Yu 2026)

---

*Full references in `mission_05_offline_rl_health.json`*
