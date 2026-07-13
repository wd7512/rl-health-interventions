# Mission 02: MRT Design & Statistical Methods (2020-2026)

## Summary

This scan identified **24 papers** (2020-2026) on micro-randomized trial methodology. The field has matured rapidly since the foundational WCLS estimator paper in 2020, with significant advances in causal inference, efficiency, missing data, sample size calculation, and integration with reinforcement learning.

## Papers Found

| # | Year | First Author | Short Title | Venue | Relevance |
|---|------|-------------|-------------|-------|-----------|
| 1 | 2020 | Qian | MRT Data Analysis Methods (WCLS) | arXiv | 5 |
| 2 | 2020 | Xu | Multi-Level MRT Design | arXiv | 4 |
| 3 | 2021 | Shi | Cluster-Level Heterogeneity in MRTs | arXiv | 4 |
| 4 | 2021 | Qian | MRT Design & Analysis Tutorial | Psychol. Methods | 5 |
| 5 | 2022 | Shi | Direct/Indirect Causal Excursion Effects | arXiv | 4 |
| 6 | 2023 | Shi | DR-WCLS Meta-Learning (Doubly Robust) | JRSS-B | 5 |
| 7 | 2023 | Shi | Auxiliary Variables for Efficiency | arXiv | 4 |
| 8 | 2023 | Bao | Per-Decision IPW for Binary Outcomes | Biostatistics | 4 |
| 9 | 2023 | Liu | Zero-Inflated Count Outcomes (Drink Less) | Biometrics | 4 |
| 10 | 2023 | Cheng | Efficient & Globally Robust CEE | JASA | 5 |
| 11 | 2024 | Huch | Data Integration for Multiple MRTs | Biometrics | 4 |
| 12 | 2024 | Yu | Time-Varying Effects (HeartSteps Functional) | arXiv | 5 |
| 13 | 2024 | Yu | Doubly Robust CEE with Missing Data | arXiv | 4 |
| 14 | 2025 | Meng | HeartSteps Online Sampling Algorithm Eval | arXiv | 5 |
| 15 | 2025 | Qian | Distal Causal Excursion Effects (DCEE) | arXiv | 5 |
| 16 | 2025 | Lin | Categorical Treatments & Sample Size | arXiv | 5 |
| 17 | 2025 | Wei | Off-Policy Inference (NSAVE, Drink Less) | arXiv | 4 |
| 18 | 2025 | Qian | Dynamic Causal Mediation for MRTs | arXiv | 4 |
| 19 | 2025 | Yu | Moderated Causal Excursion Odds Ratio | arXiv | 4 |
| 20 | 2025 | Cha | DR-EMEE with Stabilized Weights | arXiv | 4 |
| 21 | 2025 | Li | Power-Constrained Nonstationary Bandits (ROGUE-TS) | arXiv | 5 |
| 22 | 2025 | Bose | Conformal Prediction for ITEs in MRTs | arXiv | 4 |
| 23 | 2026 | Li | Hybrid SMART-MRT Designs | arXiv | 4 |
| 24 | 2026 | Abbott | Joint Model for Longitudinal/Event Outcomes | arXiv | 3 |

## Key Themes

### 1. WCLS Estimator and Causal Excursion Effects (Core Framework)
The WCLS estimator (Qian et al., 2020, 2021) provides the foundation for causal inference in MRTs. The causal excursion effect framework allows researchers to estimate time-varying marginal and moderated treatment effects. Subsequent work has extended this framework to handle cluster-level heterogeneity (Shi et al., 2021), binary outcomes (Bao et al., 2023; Yu & Qian, 2025), count data (Liu et al., 2023), categorical treatments (Lin & Qian, 2025), and distal outcomes (Qian, 2025).

### 2. Doubly Robust and Efficient Estimation
A major thrust has been improving efficiency and robustness of MRT estimators:
- **DR-WCLS** (Shi & Dempsey, 2023) handles unknown randomization, missing data, and high-dimensional features simultaneously
- **Efficient CEE** (Cheng et al., 2023) derives the semiparametric efficiency bound for CEE estimation
- **Per-decision IPW** (Bao et al., 2023) dramatically reduces variance for binary outcomes
- **DR-EMEE** (Cha & Cha, 2025) achieves 2x efficiency over standard IPW with stabilized weights

### 3. Sample Size and Power Analysis
- **Categorical treatments** (Lin & Qian, 2025): First sample size formula for multi-arm MRTs
- **Multi-level MRT** (Xu et al., 2020): GEE-based sample size calculators with R Shiny app
- Earlier foundational work (Liao et al., 2016; Seewald et al., 2016) provided the initial MRT-SS Calculator

### 4. Missing Data and Practical Challenges
- **Doubly robust missing data** (Yu & Qian, 2024): Handles MAR longitudinal outcomes in MRTs
- **HeartSteps constraints** (Meng et al., 2025): Evaluation of online sampling algorithm for randomization constraints
- **Auxiliary variables** (Shi et al., 2023): Leveraging underutilized sensor data for efficiency

### 5. Integration with Reinforcement Learning and Bandits
- **Power-constrained bandits** (Li et al., 2025): ROGUE-TS with probability clipping for MRT-compatible exploration
- **Off-policy inference** (Wei, 2025): NSAVE method for optimal policy evaluation using MRT data
- **Conformal prediction** (Bose & Dempsey, 2025): Individual treatment effect intervals using MRT simulations

### 6. New Designs and Extensions
- **Hybrid SMART-MRT** (Li et al., 2026): Combines SMART for slow adaptation with MRT for fast adaptation
- **Joint modeling** (Abbott et al., 2026): Bayesian longitudinal-survival model for complex MRT outcomes
- **Causal mediation** (Qian, 2025): Decomposes total excursion effects into direct and indirect pathways

## Negative/Null Results and Gaps

Most MRT methodology papers report positive results (new estimators with theoretical guarantees). Notable gaps and challenges identified:

1. **Underpowered MRTs**: Several applied MRTs have struggled with statistical power due to small samples and high variance. The sample size papers (Lin & Qian, 2025; Xu et al., 2020) explicitly address the risk of underpowered studies.
2. **Missing data remains pervasive**: Yu & Qian (2024) note that missingness due to sensor non-wear and missed self-reports is endemic in mHealth MRTs and can severely bias naive analyses.
3. **Treatment effect heterogeneity**: The DR-WCLS work (Shi & Dempsey, 2023) acknowledges that linear models with prespecified features often fail to capture the complexity of mobile health data.
4. **Constraint-managed randomization**: Meng et al. (2025) document the challenges of maintaining randomization constraints in HeartSteps V2V3, noting that online algorithms can introduce dependencies that complicate analysis.
5. **No published systematic review of failed MRTs**: There is a notable absence of papers cataloging null or negative results from MRTs, suggesting potential publication bias.

## Relevance to Project

Papers with **relevance = 5** are most directly applicable:
- **arXiv:2504.15484** (Lin & Qian): Categorical treatments directly applicable to 4-action MDP
- **arXiv:2502.13500** (Qian): DCEE bridges proximal MRT outcomes to long-term habit formation
- **arXiv:2511.02944** (Li et al.): Power-constrained bandits directly relevant for next steps
- **arXiv:2501.02137** (Meng et al.): HeartSteps constraint algorithm evaluation
- **arXiv:2410.15049** (Yu & Qian): Time-varying effects analysis of HeartSteps step count data
- **arXiv:2004.10241** (Qian et al.): Foundational WCLS for primary MRT analyses
- **arXiv:2306.16297** (Shi & Dempsey): DR-WCLS for robust estimation
- **arXiv:2107.03544** (Qian et al.): Comprehensive tutorial for design and analysis

## Sources Searched
- arXiv (stat.ME, stat.AP, stat.ML, cs.LG)
- PubMed/MEDLINE
- Google Scholar

## Limitations
- Google Scholar was rate-limited during the search
- PubMed returned truncated results for some queries
- Focus on English-language, peer-reviewed or arXiv-preprint literature
- Some 2026 papers are still in preprint
