---
title: "Recreation — PEARL (Lee et al., 2025)"
status: "recreation report v0.1"
date: "2026-07-13"
paper: "Lee, A. A., Hegde, N., Deliu, N., et al. (2025). A Personalized Exercise Assistant using Reinforcement Learning (PEARL): Results from a four-arm Randomized-controlled Trial. arXiv:2508.10060."
purpose: "Largest RL-for-physical-activity RCT to date; provides real-world effect-size calibration for our simulator."
related: "heartsteps-v1.md · heartsteps-v2-effect-size.md · stepcountjitai-baselines.md"
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

## 2. Headline Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Enrolled & randomized | 13,463 | Abstract |
| Primary analysis N | 7,711 (mITT) | Abstract |
| Baseline daily steps | 5,618.2 | Abstract |
| RL vs Control (1 mo) | +296 steps, p=0.0002 | Abstract |
| RL vs Random (1 mo) | +218 steps, p=0.005 | Abstract |
| RL vs Fixed (1 mo) | +238 steps, p=0.002 | Abstract |
| RL vs Control (2 mo) | +210 steps, p=0.0122 | Abstract |
| GEE sustained effect | +208 steps, p=0.002 | Abstract |
| Nudge bank size | 155 | Abstract |
| Behavioural framework | COM-B | Abstract |
| Attrition rate | 42.7% | Derived |
| Effect size (% baseline) | ~5.3% at 1 month | Derived |

## 3. Recreation Methodology

PEARL is a real-world RCT deployed through the Fitbit app. Full recreation is not possible — the algorithm is proprietary, the nudge bank is not public, and participant data is not shared.

**What we can extract:**
- Effect sizes for calibration: +200–300 steps is the real-world benchmark
- Four-arm design as an evaluation template
- COM-B as a theoretical grounding for state space design

**What we cannot extract:**
- Exact RL algorithm (proprietary to Google)
- The 155 nudge messages or their categorization
- State feature representation and reward function specifics
- Subgroup analyses and heterogeneity of treatment effects

## 4. Four-Question Loop

### 4.1 Is +296 steps at the theoretical limit?

The +296 step increase is ~5.3% of baseline — below the MCID of ~1,000 steps/day for individuals, but meaningful at population scale.

For comparison:
- HeartSteps V1: ATE +35 steps (p=0.06)
- HeartSteps V2: 78.4% improved, +29.75 mean improvement
- PEARL: +296 steps (p=0.0002)

The theoretical limit for a contextual bandit is `O(√(dT log T))` (Russo & Van Roy 2018). With 155 actions and ~60 decision points per participant, the algorithm is severely data-limited. PEARL's effect size suggests real signal but far from saturating the action space.

**Assessment:** Not at the theoretical limit. A simulator with known transitions could test whether richer state representations push effect sizes higher.

### 4.2 How does this link to other papers and this work?

| Paper / PR | Connection |
|------------|------------|
| HeartSteps V1 (Klasnja 2019) | Predecessor MRT; PEARL scales from ~50 to 13,463 |
| HeartSteps V2 (Klasnja 2022) | Same algorithmic family; PEARL scales action space from ~5 to 155 |
| StepCountJITAI (Karine 2024) | Simulation-based; PEARL validates real-world effect size range |
| COM-B (Michie 2011) | PEARL's behavioural framework; our project does not yet integrate one |
| Our project | PEARL provides effect-size calibration target |

### 4.3 Is there still room for improvement?

1. **Longer horizons:** PEARL only reports 2-month outcomes. Our simulator can run 90–9000 day episodes.
2. **Burden-aware rewards:** PEARL does not model notification fatigue. Our reward includes a burden penalty.
3. **Counterfactual analysis:** PEARL cannot answer "what if we had used algorithm X?" Our simulator enables exact counterfactual comparison.

### 4.4 Take action

1. **Calibrate simulator effect sizes** — Target 200–300 step deltas to match PEARL.
2. **Add a fixed-rule baseline agent** — Complete the four-arm evaluation template.
3. **Cite PEARL as the scale benchmark** — In slides and design document.
4. **Test action space scaling** — PEARL uses 155 nudges vs our 4 actions.
5. **Validate persona differentiation** — Our 5 personas should show RL > Fixed > Random ordering.

## 5. Validation Assessment

**What is validated:**
- RL-personalized nudges produce significant step count increases vs all comparators at scale
- Effect sustained at 2 months (attenuated), suggesting real but modest behaviour change
- Four-arm design cleanly isolates personalization value

**What is not validated:**
- Mechanism of action — which COM-B components drive the effect
- Long-term sustainment beyond 2 months
- Generalizability — 86.3% female, Fitbit users only

**Why this matters:** PEARL validates that RL for PA works in the real world, but effects are modest (~5% of baseline). Our simulator should produce comparable effect sizes to be credible.

## 6. Open Questions

- [ ] What is the exact RL algorithm? (Thompson Sampling? LinUCB?)
- [ ] How were 155 nudges categorized by COM-B component?
- [ ] What is the heterogeneity of treatment effects across user types?
- [ ] Does the RL arm show learning over time, or is the effect constant?
- [ ] How does PEARL's +296 steps compare to our simulator's effect sizes?
- [ ] Could we implement a PEARL-style fixed arm as a baseline?
- [ ] What is the notification burden (nudges/day) in PEARL?

## 7. Citation

```bibtex
@misc{lee2025pearl,
  author  = {Lee, Amy Armento and Hegde, Narayan and Deliu, Nina and Rosenzweig, Emily
             and Suggala, Arun and Lakshminarasimhan, Sriram and He, Qian
             and Hernandez, John and Seneviratne, Martin and Singh, Rahul
             and Kalkar, Pradnesh and Shanmugam, Karthikeyan and Raghuveer, Aravindan
             and Singh, Abhimanyu and Nguyen, My and Taylor, James and Alla, Jatin
             and Villar, Sofia S. and Emir-Farinas, Hulya},
  title   = {A Personalized Exercise Assistant using Reinforcement Learning ({PEARL}):
             Results from a four-arm Randomized-controlled Trial},
  year    = {2025},
  month   = {August},
  eprint  = {2508.10060},
  archiveprefix = {arXiv},
  primaryclass = {cs.LG},
  doi     = {10.48550/arXiv.2508.10060}
}
```

## 8. Next Actions

1. **Add `lee2025pearl` to `references.bib`**
2. **Include PEARL in sprint 1 slides** — "Existing Work" slide as largest RL-for-PA RCT
3. **Create PEARL-style evaluation template** — Four-arm: idle, random, fixed-rule, adaptive
4. **Calibrate transition tables** — Compare simulator deltas against +200–300 step range
5. **Write PEARL-to-simulator mapping** — Map study design elements to config parameters

---

*End of Recreation — PEARL (Lee et al., 2025) v0.1*
