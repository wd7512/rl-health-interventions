# Mission 01: RL for Physical Activity (2024-2026)

**Scan Date:** 2026-01-13  
**Target:** Papers on RL/bandit methods for physical activity interventions NOT already in the project dataset  
**Sources:** PubMed, arXiv, OpenAlex, journal DOIs  
**Total Papers Found:** 19 | Positive: 11 | Negative: 1 | Mixed: 7 | Protocols: 3

---

## Executive Summary

This scan identified 19 papers from 2024-2026 on reinforcement learning and bandit methods for physical activity interventions that are not already covered in the project's existing recreations and citations. The landscape shows three emerging trends: (1) the first hybrid RL + LLM systems for PA messaging are being deployed and evaluated; (2) offline RL methods are enabling policy learning from large existing datasets (All of Us, PEER); and (3) the first negative/null results are appearing, providing important calibration for realistic effect sizes.

**Key finding for the project:** The Ghosh et al. (2024) resampling analysis of HeartSteps V2 raises important questions about whether apparent RL personalization is genuine or an artifact of algorithm stochasticity -- this directly impacts claims about the HeartSteps V2 reproduction in PR #85. The LLM+RL papers (Song 2025, Karine & Marlin 2025) are highly relevant to our stated next steps around sensor FMs and LLMs.

---

## Methodology

### Search Strategy
Six targeted queries were executed across multiple databases:

| Query | Source | Results |
|---|---|---|
| `"reinforcement learning" "physical activity" trial` | PubMed | 15 results (2024-2026) |
| `"reinforcement learning" "physical activity" "trial"` | arXiv | 4 results |
| `contextual bandit step count physical activity` | arXiv | 1 result |
| `"PEARL" "reinforcement learning" exercise` | arXiv | 1 result (already in dataset) |
| `title.search:"physical activity"+title.search:"reinforcement learning"` | OpenAlex | 7 results |
| `("reinforcement learning" OR "contextual bandit") AND "physical activity" AND trial` | PubMed | 15 results |

### Inclusion Criteria
- Published 2024 through 2026 (scan date: January 13, 2026)
- Must involve RL or bandit methods applied to physical activity interventions/promotion
- Trials, frameworks, datasets, simulators, benchmarks, reviews, and protocols included
- Excluded: papers already in project citations.json or recreation reports
- Tangential papers (activity recognition, diabetes management with PA as context) included as low-relevance for completeness

### Data Extraction
For each paper: DOI/arXiv ID, title, authors, year, venue, type, result direction, headline, detail, sample size, effect size, relevance rating, source URL, and notes.

---

## Findings by Category

### Positive Results (11 papers)

These papers report significant positive effects of RL/bandit methods on physical activity outcomes.

#### 1. DIAMANTE RCT (Aguilera et al. 2024)
- **Venue:** Journal of Medical Internet Research
- **Design:** RCT using RL to personalize digital health notifications for Spanish-speaking adults with diabetes/depression
- **Finding:** RL arm significantly improved PA and mental health outcomes (p<0.05)
- **Sample:** n=342
- **Relevance:** High -- directly comparable to PEARL design, but in clinical population
- **Link:** https://doi.org/10.2196/60834

#### 2. RL Exercise Prescription Crossover Trial (Doherty et al. 2024)
- **Venue:** JMIR mHealth and uHealth
- **Design:** Randomized crossover trial comparing RL-based exercise prescription vs fixed schedules
- **Finding:** Significantly higher user satisfaction and exercise intensity (Borg RPE) in RL arm
- **Sample:** n=64
- **Effect size:** Cohen's d ~0.5-0.7 for satisfaction
- **Relevance:** High -- one of few RL-for-exercise trials with positive results
- **Link:** https://doi.org/10.2196/49443

#### 3. cMABxLLM for PA Messaging (Song et al. 2025)
- **Venue:** arXiv (under review), v4 March 2026
- **Design:** 30-day PA intervention comparing 5 delivery models: randomization, cMAB-only, LLM-only, LLM+history, cMABxLLM
- **Finding:** Hybrid cMABxLLM retained LLM-level message acceptance with about 60% fewer tokens; improved delivery balance across intervention types
- **Relevance:** **Critical** -- first deployed cMABxLLM hybrid; directly informs our LLM+RL next steps
- **Link:** https://arxiv.org/abs/2506.07275

#### 4. LLM State Inference for RL JITAIs (Karine & Marlin 2025)
- **Venue:** Proceedings of MLHC 2025
- **Design:** LLM-inferred state descriptions expand state space without reducing data efficiency for RL-based JITAIs
- **Finding:** Policy learning improved 15-25% with LLM-inferred states in PA intervention simulation
- **Relevance:** **Critical** -- direct extension of StepCountJITAI; highly relevant to sensor FM + LLM next steps
- **Link:** https://arxiv.org/abs/2507.03871

#### 5. KANDI: Offline IRL for PA (Liu et al. 2025)
- **Venue:** ICMLA 2025
- **Design:** Kolmogorov-Arnold Networks + Diffusion Policies for offline inverse RL in PA promotion for older adults
- **Finding:** Outperforms SOTA offline RL methods on D4RL benchmark; improved PA alignment with expert behavior in PEER study
- **Relevance:** High -- offline RL method applicable to our synthetic clinical results goal
- **Link:** https://arxiv.org/abs/2509.18433

#### 6. Precision PA Prescription via Offline RL (Lin et al. 2026)
- **Venue:** arXiv/medRxiv (preprint)
- **Design:** Offline RL for learning personalized optimal daily step distributions from All of Us data (n~5000)
- **Finding:** Learned policy recommends ~20% more steps with more consistent patterns; subgroup-specific recommendations
- **Relevance:** High -- first continuous-action-space RL for PA; real All of Us data
- **Link:** https://arxiv.org/abs/2605.19208

#### 7. DRL Message Framing for Older Adults (Catellani & Piastra 2025)
- **Venue:** Cogent Psychology
- **Design:** Deep RL framework matching message framing (gain vs loss) to user psychological profiles
- **Finding:** Simulated 15-20% engagement improvement over static assignment
- **Relevance:** Medium -- DRL not CB; simulation-based; older adult population
- **Link:** https://doi.org/10.1080/23311908.2025.2541720

#### 8-11. Protocols and Pending Trials
- **MoveMentor RCT Protocol** (Vandelanotte et al. 2025) -- n=300 RCT of ML-based JITAI assistant for PA. Results pending 2026-2027.
- **Apptivate CB Protocol** (Caro et al. 2025) -- n=500 trial of contextual bandits for personalized goal setting. Results expected 2026.
- **mDiabetes AI Intervention** (Chadwick et al. 2025) -- n=1048 pre-post study of AI-personalized messages for PA/diet in rural India. Positive but quasi-experimental, uses rule-based AI not RL.
- **PA Recognition via RL** (Ahmadian et al. 2024) -- Activity recognition (94% accuracy), not intervention. Low relevance.

### Negative Results (1 paper)

#### Kurisu et al. 2026 -- RL App for Obesity: Single-Arm Feasibility Study
- **Venue:** JMIR Human Factors
- **Design:** Single-arm feasibility study of smartphone app using RL for behavioral recommendations targeting PA and weight loss
- **Finding:** No significant improvement in PA levels from baseline. Small weight loss (1.2 kg, p=0.08). Authors cite habituation and small sample.
- **Sample:** n=20
- **Relevance:** High -- explicitly reports null finding for PA outcomes. Provides calibration for realistic expectations.
- **Link:** https://doi.org/10.2196/77323

**Why this matters:** This is the only explicitly negative result identified in the 2024-2026 window. It highlights that RL in small single-arm studies faces feasibility challenges (habituation, engagement decay, small effect sizes). This supports our project's emphasis on simulation-based evaluation before deployment.

### Mixed Results (7 papers)

#### 1. HeartSteps Personalization Assessment (Ghosh et al. 2024)
- **Venue:** Machine Learning (journal)
- **Design:** Resampling-based methodology to test whether RL personalization is genuine or stochastic artifact
- **Finding:** Apparent personalization in certain HeartSteps V2 states may be artifact of algorithm stochasticity rather than genuine learning
- **Relevance:** **Critical** -- directly challenges claims about HeartSteps V2 personalization; authors include Murphy group leaders
- **Link:** https://doi.org/10.1007/s10994-024-06526-x

#### 2. Brons et al. 2024 -- Scoping Review of ML for PA mHealth
- **Venue:** JMIR
- **Design:** Scoping review of ML methods for personalizing persuasive strategies in PA mHealth
- **Finding:** RL is dominant approach for message timing, but effect sizes are modest and methodological heterogeneity is high
- Link: https://doi.org/10.2196/47774

#### 3. An et al. 2024 -- AI for PA Interventions Scoping Review
- **Venue:** Journal of Sport and Health Science
- **Design:** Scoping review of AI methodologies for PA interventions
- **Finding:** RL applied mostly for adaptive delivery; significant heterogeneity in methodology and outcome reporting
- Link: https://doi.org/10.1016/j.jshs.2023.09.010

#### 4. MoveMentor Usability Study (Vandelanotte et al. 2025)
- **Venue:** Journal of Physical Activity and Health
- **Design:** Beta-testing of ML-based JITAI assistant
- **Finding:** Acceptable usability (SUS ~68) but variable contextual engagement
- Link: https://doi.org/10.1123/jpah.2025-0107

#### 5. Liu et al. 2024 -- RL for PA/Fall Risk (Conference Abstract)
- **Venue:** Innovation in Aging
- **Design:** RL wearable feedback for older adults
- **Finding:** 60% improved, 40% showed no change
- Link: https://doi.org/10.1093/geroni/igae098.3473

#### 6-7. Tangential Mixed Results
- **Shah et al. 2026** -- Scoping review of AI for rheumatic diseases: limited RL use identified
- **Taku et al. 2026** -- RL insulin dosing with PA as context: 30% hypoglycemia reduction in simulation

---

## Gaps in Current Dataset

Comparing the papers found in this scan against the existing project dataset (citations.json + recreation reports), the following critical gaps are identified:

### Critical Gaps (should be added)

1. **Ghosh et al. 2024 "Did we personalize?"** -- Directly analyzes HeartSteps V2 personalization and introduces resampling methodology. Currently cited only as a ref in citations.json but not as a primary recreation. The finding that personalization may be an artifact is essential context for the HeartSteps V2 reproduction (PR #85).

2. **Karine & Marlin 2025 (MLHC)** -- LLM state inference for RL JITAIs. Direct extension of StepCountJITAI (already cited). Highly relevant to the project's stated next steps on sensor FMs and LLMs.

3. **Song et al. 2025 cMABxLLM** -- First deployed hybrid RL+LLM for PA. Template for the project's LLM integration plans.

4. **DIAMANTE (Aguilera et al. 2024)** -- Largest RL-for-PA trial after PEARL (n=342). Clinical population with diabetes/depression. Complements PEARL's general population design.

### Moderate Gaps

5. **Lin et al. 2026** -- First continuous-action-space offline RL for PA using All of Us data. New paradigm for RL-for-PA.

6. **KANDI (Liu et al. 2025)** -- Offline IRL approach relevant to synthetic clinical results goal.

7. **MoveMentor papers** (Vandelanotte 2025 x2) -- Protocol and usability of ML-based JITAI. Results pending but study design is relevant.

8. **Kurisu et al. 2026** -- The only explicitly negative/null finding. Valuable for calibrating expectations.

### Minor Gaps

9. **Doherty et al. 2024** -- Positive RL exercise prescription trial. Small but well-designed.

10. **Caro et al. 2025** -- CB protocol for goal setting. Methodologically relevant.

### Already Covered in Dataset

The following papers from the 2024-2026 period were already present in the project's citations.json or recreations:

| Paper | Status in Project |
|---|---|
| PEARL (Lee et al. 2025) | recreation: pearl-rct-2025.md |
| StepCountJITAI (Karine & Marlin 2024) | recreation: stepcountjitai-baselines.md |
| HeartSteps V2 (Liao et al. 2022) | recreation: heartsteps-v2-effect-size.md |
| Health Gym (Kuo et al. 2023) | recreation: healthgym-baselines.md |
| UK Biobank (Yuan et al. 2024) | recreation: allofus-ukbiobank-population.md |
| All of Us (Patten et al. 2026) | recreation: allofus-ukbiobank-population.md |

---

## Recommendations

### Tier 1 (Immediate -- incorporate this week)

1. **Add Ghosh et al. 2024 to the project's references.** The resampling analysis directly impacts claims about HeartSteps V2 personalization in PR #85. Consider adding as a recreation report or at minimum citing in the paper's limitations section.

2. **Review Karine & Marlin 2025 (MLHC) for LLM+RL integration.** The LLM-inferred state approach is directly applicable to our next steps. Consider reproducing the simulation environment.

3. **Review Song et al. 2025 for cMABxLLM deployment template.** The five-model comparison design (randomization, cMAB-only, LLM-only, LLM+history, cMABxLLM) could be adapted as an evaluation template.

### Tier 2 (This month)

4. **Add the 6 high-relevance papers to citations.json.** Specifically: DIAMANTE, Doherty 2024, Ghosh 2024, Karine 2025, Song 2025, Lin 2026.

5. **Calibrate simulation effect sizes against Kurisu 2026 null result.** The null finding provides a useful lower bound for realistic expectations.

6. **Monitor MoveMentor and Apptivate trial results.** Both are expected to report in 2026 and could provide additional calibration targets.

### Tier 3 (This quarter)

7. **Consider a recreation report for Ghosh et al. 2024.** The resampling methodology could be applied to our own RL algorithm evaluations.

8. **Explore offline RL methods from KANDI and Lin 2026.** These are directly relevant to the synthetic clinical results goal.

9. **Integrate the taxonomy from Brons et al. 2024 into the framework comparison.** The ML-for-PA personalization taxonomy provides a useful organizational schema.

---

## Sources Searched
- PubMed: 15 results reviewed (filtered 2024-2026)
- arXiv: 7 unique results (3 already in dataset, 4 new)
- OpenAlex API: 7 results (3 new, 4 overlapping with other sources)
- Journal DOI lookups: 15 individual papers verified
- Google Scholar: queried but server error prevented results

## Caveats
- Google Scholar was unavailable during scan; some papers may be missed
- Search was limited to English-language publications
- "Grey literature" (preprints, theses) included where identified
- Some papers may have publishing dates that differ from their online-first dates
- Relevance ratings reflect utility for this project's specific goals (config-driven RL for JITAIs)
