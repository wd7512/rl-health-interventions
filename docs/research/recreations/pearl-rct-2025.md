---
title: "Recreation — PEARL (Lee et al., 2025)"
status: "recreation report v0.2"
date: "2026-07-21"
paper: "Lee, A. A., Hegde, N., Deliu, N., et al. (2025). A Personalized Exercise Assistant using Reinforcement Learning (PEARL): Results from a four-arm Randomized-controlled Trial. arXiv:2508.10060."
doi_arxiv: "10.48550/arXiv.2508.10060"
doi_jmir: "10.2196/preprints.91156"
clinical_trial: "OSF.IO/TW7UP (doi: 10.17605/OSF.IO/TW7UP)"
license: "CC BY-NC-SA 4.0"
publisher: "JMIR Publications Inc."
purpose: "First large-scale RL-for-physical-activity RCT; provides real-world effect-size calibration for our simulator."
related: "heartsteps-v1.md · heartsteps-v2-effect-size.md · stepcountjitai-baselines.md"
openalex_id: "W7125932538"
openalex_topics:
  - "Digital Mental Health Interventions"
  - "Physical Activity and Health"
  - "Behavioral Health and Interventions"
---

# Recreation — PEARL (Lee et al., 2025)

## 1. Paper Overview

PEARL is the first large-scale, four-arm RCT assessing an RL algorithm — informed by the COM-B behaviour change framework — to personalize content and timing of physical activity nudges via a Fitbit app.

**Study design:** 13,463 Fitbit users randomized into four arms:

| Arm | Description |
|-----|-------------|
| Control | No nudges |
| Random | Nudges selected uniformly at random from 155 |
| Fixed | Nudges selected by pre-set logic from self-reported PA barriers |
| RL | Nudges selected by an adaptive contextual bandit algorithm |

**Primary analysis:** 7,711 participants (mITT; mean age 42.1, 86.3% female, baseline 5,618 steps/day).

**Key finding:** RL arm significantly outperformed all three comparators at 1 month, with sustained effects at 2 months.

## 2. Full Author List

| # | Author | ORCID |
|---|--------|-------|
| 1 | Amy Armento Lee (first) | [0000-0002-7564-7891](https://orcid.org/0000-0002-7564-7891) |
| 2 | Narayan Hegde | — |
| 3 | Nina Deliu | [0000-0003-2501-8795](https://orcid.org/0000-0003-2501-8795) |
| 4 | Emily Rosenzweig | — |
| 5 | Arun Suggala | [0000-0003-4113-5924](https://orcid.org/0000-0003-4113-5924) |
| 6 | Sriram Lakshminarasimhan | — |
| 7 | Qian He | — |
| 8 | John Hernandez | — |
| 9 | Martin Seneviratne | — |
| 10 | Rahul Singh | — |
| 11 | Pradnesh Kalkar | — |
| 12 | Karthikeyan Shanmugam | — |
| 13 | Aravindan Raghuveer | [0000-0001-5006-4385](https://orcid.org/0000-0001-5006-4385) |
| 14 | Abhimanyu Singh | — |
| 15 | Hariharan Manoharan | — |
| 16 | My Nguyen | — |
| 17 | James Taylor | — |
| 18 | Jatin Alla | — |
| 19 | Sofia S. Villar | — |
| 20 | Hulya Emir-Farinas (last) | — |

## 3. Publication Metadata

| Field | Value |
|-------|-------|
| arXiv | [2508.10060](https://arxiv.org/abs/2508.10060) (submitted Aug 12, 2025) |
| JMIR Preprint | [10.2196/preprints.91156](https://doi.org/10.2196/preprints.91156) (Jan 27, 2026) |
| Clinical Trial | [OSF.IO/TW7UP](https://doi.org/10.17605/OSF.IO/TW7UP) |
| License | CC BY-NC-SA 4.0 |
| Publisher | JMIR Publications Inc. |
| OpenAlex ID | [W7125932538](https://openalex.org/W7125932538) |
| References | 27 cited works (26 unique DOIs) |
| OpenAlex Topics | Digital Mental Health Interventions; Physical Activity and Health; Behavioral Health and Interventions |
| OpenAccess | Gold OA |

## 4. Headline Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Enrolled & randomized | 13,463 | Abstract |
| Primary analysis N | 7,711 (mITT) | Abstract |
| Mean age | 42.1 years | Abstract |
| Female | 86.3% | Abstract |
| Baseline daily steps | 5,618.2 | Abstract |
| RL vs Control (1 mo) | +296 steps, p=0.0002 | Abstract |
| RL vs Random (1 mo) | +218 steps, p=0.005 | Abstract |
| RL vs Fixed (1 mo) | +238 steps, p=0.002 | Abstract |
| RL vs Control (2 mo) | +210 steps, p=0.0122 | Abstract |
| GEE sustained effect (RL vs Control) | +208 steps, p=0.002 | Abstract |
| Nudge bank size | 155 | Abstract |
| Behavioural framework | COM-B | Abstract |
| Randomized-to-analysis reduction | 42.7% | Derived (13,463 - 7,711) / 13,463 |
| Effect size (% baseline) | ~5.3% at 1 month | Derived |

## 5. Full Abstract

> **BACKGROUND:** Consistent physical inactivity among adults and adolescents poses a major global health challenge. Mobile health (mHealth) interventions, particularly Just-in-Time Adaptive Interventions (JITAIs), offer a promising avenue for scalable and personalized physical activity promotion. However, developing and evaluating such adaptive interventions at scale, while integrating robust behavioral science, presents methodological hurdles.
>
> **OBJECTIVE:** The PEARL study aimed to assess the feasibility and effectiveness of a reinforcement learning (RL) algorithm, informed by health behavior change theory (COM-B), to personalize the content and timing of physical activity nudges via the Fitbit app compared to fixed and random nudging strategies, and to a control group with no nudges.
>
> **METHODS:** We conducted a large-scale, four-arm randomized controlled trial (RCT) enrolling 13,463 Fitbit users. Participants were randomized to: (1) Control (no nudges); (2) Random (random content/timing); (3) Fixed (logic based on baseline COM-B survey); and (4) RL (adaptive algorithm). The primary outcome was the change in average daily step count from baseline to 2 months. Secondary outcomes included user engagement and survey responses regarding capability, opportunity, and motivation.
>
> **RESULTS:** 7,711 participants were included in the primary analysis (mean age 42.1 years; 86.3% female). At 1 month, the RL group showed a significant increase in daily steps compared to Control (+296 steps, P<.001), Random (+218 steps, P=.005), and Fixed (+238 steps, P=.002) groups. At 2 months, the RL group sustained a significant increase against the Control (+210 steps, P=.01). Generalized estimating equation (GEE) models confirmed a sustained significant increase in the RL group (+208 steps, P=.002). In exit surveys, the RL group reported higher favorable responses regarding nudge customization (37%) compared to other groups.
>
> **CONCLUSIONS:** This study demonstrates the feasibility and early efficacy of using RL to personalize digital health nudges at scale. While long-term retention remains a challenge, the adaptive approach outperformed static behavioral rules, showcasing the promise of dynamic personalization in a real-world mHealth setting.

## 6. Recreation Methodology

PEARL is a real-world RCT deployed through the Fitbit app. Full recreation is not possible — the algorithm is proprietary, the nudge bank is not public, and participant data is not shared.

**What we can extract:**
- Effect sizes for calibration: +200-300 steps is the real-world benchmark
- Four-arm design as an evaluation template
- COM-B as a theoretical grounding for state space design

**What we cannot extract:**
- Exact RL algorithm (proprietary to Google)
- The 155 nudge messages or their categorization
- State feature representation and reward function specifics
- Subgroup analyses and heterogeneity of treatment effects

## 7. Four-Question Loop

### 7.1 Is +296 steps at the theoretical limit?

The +296 step increase is ~5.3% of baseline — below the MCID of ~1,000 steps/day for individuals, but meaningful at population scale.

For comparison:
- HeartSteps V1: ATE +35 steps (p=0.06)
- HeartSteps V2: 78.4% improved, +29.75 mean improvement
- PEARL: +296 steps (p=0.0002)

The theoretical regret bound for a contextual bandit is `O(sqrt(dT log T))` (Russo & Van Roy 2018), where `d` is the feature dimension and `T` is the total number of decision points. With 155 actions and ~60 decision points per participant (~460K pooled across 7,711 participants), the algorithm is severely data-limited relative to the action space. PEARL's effect size suggests real signal but far from saturating the action space.

**Assessment:** Not at the theoretical limit. A simulator with known transitions could test whether richer state representations push effect sizes higher.

### 7.2 How does this link to other papers and this work?

| Paper / PR | Connection |
|------------|------------|
| HeartSteps V1 (Klasnja 2019) | Predecessor MRT; PEARL scales from ~50 to 13,463 |
| HeartSteps V2 (Klasnja 2022) | Same algorithmic family; PEARL scales action space from ~5 to 155 |
| StepCountJITAI (Karine 2024) | Simulation-based; PEARL validates real-world effect size range |
| COM-B (Michie 2011) | PEARL's behavioural framework; our project does not yet integrate one |
| Our project | PEARL provides effect-size calibration target |

### 7.3 Is there still room for improvement?

1. **Longer horizons:** PEARL only reports 2-month outcomes. Our simulator can run 90-9000 day episodes.
2. **Burden-aware rewards:** PEARL does not model notification fatigue. Our reward includes a burden penalty.
3. **Counterfactual analysis:** PEARL cannot answer "what if we had used algorithm X?" Our simulator enables exact counterfactual comparison.

### 7.4 Take action

1. **Calibrate simulator effect sizes** — Target 200-300 step deltas to match PEARL.
2. **Add a fixed-rule baseline agent** — Complete the four-arm evaluation template.
3. **Cite PEARL as the scale benchmark** — In slides and design document.
4. **Test action space scaling** — PEARL uses 155 nudges vs our 4 actions.
5. **Validate persona differentiation** — Our 5 personas should show RL > Fixed > Random ordering.

## 8. Validation Assessment

**What is validated:**
- RL-personalized nudges produce significant step count increases vs all comparators at scale
- Effect sustained at 2 months (attenuated), suggesting real but modest behaviour change
- Four-arm design cleanly isolates personalization value

**What is not validated:**
- Mechanism of action — which COM-B components drive the effect
- Long-term sustainment beyond 2 months
- Generalizability — 86.3% female, Fitbit users only

**Why this matters:** PEARL validates that RL for PA works in the real world, but effects are modest (~5% of baseline). Our simulator should produce comparable effect sizes to be credible.

## 9. Open Questions

- [ ] What is the exact RL algorithm? (Thompson Sampling? LinUCB?)
- [ ] How were 155 nudges categorized by COM-B component?
- [ ] What is the heterogeneity of treatment effects across user types?
- [ ] Does the RL arm show learning over time, or is the effect constant?
- [ ] How does PEARL's +296 steps compare to our simulator's effect sizes?
- [ ] Could we implement a PEARL-style fixed arm as a baseline?
- [ ] What is the notification burden (nudges/day) in PEARL?

## 10. References Cited by PEARL

The paper cites 27 works. Key references extracted from Crossref:

| Ref | DOI | Topic |
|-----|-----|-------|
| ref4 | `10.2196/23954` | JMIR mHealth/uHealth |
| ref5 | `10.1007/s12160-016-9830-8` | Annals of Behavioral Medicine |
| ref6 | `10.3390/ijerph19042267` | Int J Environ Res Public Health |
| ref7 | `10.1093/abm/kay067` | Annals of Behavioral Medicine |
| ref8 | `10.2196/jmir.7994` | J Med Internet Res |
| ref9 | `10.2196/mhealth.9117` | JMIR mHealth uHealth |
| ref11 | `10.2196/60834` | JMIR |
| ref12 | `10.1093/abm/kaab028` | Annals of Behavioral Medicine |
| ref13 | `10.2196/49443` | JMIR |
| ref14 | `10.1038/s41746-024-01028-5` | npj Digital Medicine |
| ref15 | `10.1038/s41746-023-00921-9` | npj Digital Medicine |
| ref16 | `10.1186/1748-5908-6-42` | Implementation Science |
| ref17 | `10.1186/1748-5908-7-37` | Implementation Science |
| ref18 | `10.1016/j.pecinn.2023.100209` | Patient Education and Counseling |
| ref19 | `10.1016/j.puhe.2015.12.009` | Public Health |
| ref20 | `10.2105/ajph.2022.307150` | Am J Public Health |
| ref21 | `10.1109/tnn.1998.712192` | IEEE Trans Neural Networks |
| ref23 | `10.1111/insr.12583` | International Statistical Review |
| ref24 | `10.1037/hea0000306` | Health Psychology |
| ref26 | `10.1016/j.hrthm.2020.02.013` | Heart Rhythm |
| ref27 | `10.1007/s00415-016-8334-6` | Journal of Neurology |
| ref28 | `10.3386/w4509` | NBER Working Paper |
| ref29 | `10.1177/1094428104263672` | Group Dynamics |
| ref30 | `10.1093/biomet/73.1.13` | Biometrika |
| ref31 | `10.1038/s41591-022-02100-x` | Nature Medicine |
| ref32 | `10.1038/s41591-022-02012-w` | Nature Medicine |

## 11. Citation

```bibtex
@misc{lee2025pearl,
  author  = {Lee, Amy Armento and Hegde, Narayan and Deliu, Nina and Rosenzweig, Emily
             and Suggala, Arun and Lakshminarasimhan, Sriram and He, Qian
             and Hernandez, John and Seneviratne, Martin and Singh, Rahul
             and Kalkar, Pradnesh and Shanmugam, Karthikeyan and Raghuveer, Aravindan
             and Singh, Abhimanyu and Manoharan, Hariharan and Nguyen, My
             and Taylor, James and Alla, Jatin and Villar, Sofia S.
             and Emir-Farinas, Hulya},
  title   = {A Personalized Exercise Assistant using Reinforcement Learning ({PEARL}):
             Results from a four-arm Randomized-controlled Trial},
  year    = {2025},
  month   = {August},
  eprint  = {2508.10060},
  archiveprefix = {arXiv},
  primaryclass = {cs.LG},
  doi     = {10.48550/arXiv.2508.10060},
  note    = {Also published as JMIR Preprint: 10.2196/preprints.91156}
}
```

## 12. Next Actions

1. **Add `lee2025pearl` to `references.bib`**
2. **Include PEARL in sprint 1 slides** — "Existing Work" slide as largest RL-for-PA RCT
3. **Create PEARL-style evaluation template** — Four-arm: idle, random, fixed-rule, adaptive
4. **Calibrate transition tables** — Compare simulator deltas against +200-300 step range
5. **Write PEARL-to-simulator mapping** — Map study design elements to config parameters

---

*End of Recreation — PEARL (Lee et al., 2025) v0.2 — Updated 2026-07-21 with full author list, JMIR DOI, clinical trial registration, and 27 reference DOIs from Crossref/OpenAlex.*
