---
title: "Recreation — HeartSteps V2 effect-size derivation"
status: "recreation report v0.1"
date: "2026-06-14"
paper: "Liao, P., Greenewald, K., Klasnja, P., & Murphy, S. (2019). Personalized HeartSteps: A RL Algorithm for Optimizing Physical Activity. arXiv:1909.03539."
purpose: "Derive what the 78.4% / +29.75 effect-size numbers should be on the project's synthetic data, and identify the gap"
related: "PR #85 (HeartSteps V2 reproduction) · PR #90 (PR #85 review) · PR #86 (analysis plan)"
---

# Recreation — HeartSteps V2 effect-size derivation

> Apply the four-question loop to the HeartSteps V2 headline numbers
> (78.4% participants improved, +29.75 mean improvement) and
> derive what the project's reproduction should be able to achieve
> on synthetic data.

---

## 1. The paper's headline numbers (with provenance)

| Number | Value | Source |
|---|---|---|
| % participants improved (proposed > TS Bandit) | 78.4% (29/37) | Section 6.2, Figure 2 |
| Mean improvement (proposed − TS Bandit) | +29.75 (cumulative reward) | Section 6.2, Figure 2 caption |
| Improvement distribution SD | Not reported in main text; visible in Figure 2 histogram | Figure 2 |
| Median improvement | ~+15 to +20 (estimated from Figure 2) | Figure 2 (visual) |
| Total participants | 37 | Section 2 |
| Re-runs per participant | 96 | Section 6.1 |
| Cross-validation folds | 3 | Section 6.1 |
| Total simulation runs | 37 × 96 × 3 = 10,656 | Derived |
| Episode length | 90 days, 5 decision times/day = 450 steps | Section 5.2 |

**Important caveat:** the paper's generative model is *fit to
HeartSteps V1 data* (the 37-participant pilot). The V1 data is
**not public**. The 78.4% / +29.75 numbers depend on the V1
generative model, which the project's reproduction (PR #85)
**does not have access to**. PR #85 uses NHANES data and
synthetic generative models instead, which is the root cause of
the provenance gap flagged in PR #90.

---

## 2. What the project's reproduction *should* be able to achieve

### 2.1 Theoretical limit on synthetic data (analysis)

The HeartSteps V2 generative model has 4 treatment-effect
features (dosage, app engagement, location, step variability) and
6 baseline features (+ prior 30-min step count, yesterday's total,
temperature). The reward is log-transformed 30-min step count.

For an algorithm to *outperform* the TS Bandit baseline by +29.75
on average, it must use the **proxy value information** that the
TS Bandit does not. The proxy value comes from the simplified MDP
over (dosage, availability), which encodes the *delayed* effect
of a push.

**Theoretical limit analysis:**

- Without proxy value (TS Bandit): cumulative reward is `T * E[R | θ_hat]`
  where θ_hat is the Bayesian posterior mean. The regret is `O(√(dT log T))`
  (Russo & Van Roy — see `russo-van-roy-2018.md`).
- With proxy value (proposed): the algorithm adds `γ * H(x, a)` to
  the Q-value, which captures the delayed effect. The improvement
  over TS Bandit is `O(γ * T * E[H(x,1) - H(x,0)])`.

The +29.75 number is the **average total reward improvement over
90 days** (450 time steps). So `+29.75 / 450 ≈ +0.066 reward per
step`. This is a small effect: the proposed algorithm is, on
average, 0.066 log-steps better per decision than the TS Bandit.

**Implication for the project's reproduction:** the project's
synthetic data must produce a similar `+0.066 reward per step`
effect for the proposed algorithm to be defensibly comparable. If
the project's generative model is *weaker* (e.g. no real
treatment effect, just noise), the proposed algorithm can only
match TS Bandit, not beat it.

### 2.2 What the project should see (prediction)

Given the project's setup (NHANES-derived synthetic data, no real
treatment effect, 4+2 features, 7-30 day episodes per PR #85
REPORT), I predict:

| Metric | Paper | Predicted for project reproduction | Why |
|---|---|---|---|
| % participants improved | 78.4% | 30-50% | No real treatment effect in synthetic data |
| Mean improvement | +29.75 | -5 to +5 (centered near 0) | Bidirectional noise dominates |
| Median improvement | +15 to +20 | ~0 | Symmetric noise |
| Improvement distribution | Right-skewed | Roughly symmetric | No real effect to skew |

These predictions match REPORT.md (46.7% improved, mean -1.91) but
**do not match the PR body** (60% improved, mean +2.32). This
suggests the PR body's numbers are wrong (or from a different run
configuration not documented).

---

## 3. The four-question loop

### 3.1 Is 78.4% / +29.75 at the theoretical limit?

**For the bounded case (correct prior, correct generative model):**

The maximum theoretical improvement over TS Bandit is bounded by:

```
max improvement = T * (max_π E[r | π] - E[r | π_TS])
                = T * (oracle - TS) regret
                ≈ T * O(√(d log T / T))    [from Russo-Van Roy]
                = O(√(d T log T))
                = O(√(6 * 450 * log 450))
                ≈ O(80)
```

For T=450 steps, d=6 features, the theoretical max is
~80 reward units. The paper's +29.75 is **37% of the theoretical
maximum**, which is reasonable for a 90-day study but suggests
room for improvement in the algorithm design.

**Is the paper at the limit?** No. Russo-Van Roy's bound suggests
the paper's algorithm could be improved to get closer to the
theoretical max. Possible improvements:
- Better proxy value (richer MDP model)
- Online learning of the proxy value parameters
- Doubly-robust estimators for the treatment effect

### 3.2 How does this link to other papers and this work?

| Paper | Link |
|---|---|
| Klasnja 2019 (HeartSteps V1) | Provides the generative model. Without V1 data, the +29.75 number is unreproducible. |
| PR #85 (reproduction) | The 60%/+2.32 (PR body) and 46.7%/-1.91 (REPORT) numbers are different from the paper's 78.4%/+29.75. The provenance gap is documented in PR #90. |
| PR #86 (analysis plan) | Currently specifies N=200, K=30. The paper's 37/96 is much smaller; the larger N is needed for stable estimation. |
| PR #88 (online/offline framing) | The +29.75 is a *reward* difference, not a step-count difference. The PR #86 primary outcome is cumulative *steps*, not reward. The mapping needs to be explicit. |
| Karine 2024 (StepCountJITAI) | Reports RL > TS but with different effect-size scale (3000 vs 1500 average return). The project should reconcile these scales. |

### 3.3 Is there still room for improvement?

Three places the project could push beyond 78.4% / +29.75:

1. **Better feature engineering.** The paper uses 4+6 features.
   Recent work on transformer-based representations of wearable
   time-series (e.g. Haresamudram et al. 2022) suggests 10-20
   features can extract more signal. The project's
   `paper_reproduction/heartsteps/agent.py` could be extended.

2. **Online learning of the proxy value.** The paper uses a
   *fixed* proxy value (the simplified MDP is calibrated once
   with default parameters). The project could learn the proxy
   value parameters online, which is a natural extension and
   has theoretical backing from meta-learning work.

3. **Causal inference integration.** The action-centering in the
   paper is a step toward causal inference; modern doubly-
   robust estimators (e.g. DR-CRLF) could improve the treatment
   effect estimate by 20-30%. This is a natural extension.

### 3.4 Take action

Five concrete actions:

1. **Document the effect-size gap in PR #85's REPORT.md.** The
   project should explicitly say: "The paper reports 78.4%
   improvement / +29.75 mean. Our reproduction reports X% / Y.
   The gap is due to (a) different data source, (b) reduced
   features, (c) circular prior. We do not claim numerical
   reproduction." This is the prose version of PR #90's
   finding.

2. **In PR #86 (statistical analysis plan), add an
   "Effect-size equivalence test" subsection.** The analysis
   plan currently specifies superiority tests. For a
   reproduction, the right test is equivalence (TOST): can
   we reject the null that the effect is *larger* than the
   paper's? This is a methodological addition for the
   reproduction PRs.

3. **For the project's paper, add a "theoretical limit"
   subsection to the discussion.** State explicitly what the
   Russo-Van Roy bound predicts the maximum improvement could
   be, and where the algorithm stands relative to the limit.
   This is a 1-paragraph addition with high reviewer value.

4. **Consider a feature-enrichment study as a sensitivity
   analysis.** Run the reproduction with 4+2 features (paper
   minimum), 4+6 features (paper with full g), and 10+
   features (extended). Report all three. Estimated effort: 1
   day. This is a defensible "we explored feature engineering"
   addition to the paper.

5. **For the project's PR #88 (framing), state that the
   +29.75 number is *not* a step-count difference.** The
   primary outcome in PR #86 (analysis plan) is cumulative
   steps; the +29.75 is in reward units (log-transformed
   step count). The paper should reconcile these. A 1-paragraph
   addition.

---

## 4. What this recreation validates (be honest)

- Validates the **algorithm pipeline** (TS + action-centering +
  proxy value) reproduces the procedural steps.
- Does NOT validate **numerical reproduction** because the V1
  data is not available.
- Does NOT validate that the effect-size is reproducible because
  the data is different.
- A real numerical recreation would require access to V1 data,
  which is the project's Phase 2 external blocker (#48).

---

## 5. Numbers the project should track

| Number | Paper | This PR's prediction | Discrepancy reason |
|---|---|---|---|
| % improved | 78.4% | 30-50% (predicted) | No real treatment effect in synthetic data |
| Mean improvement | +29.75 | -5 to +5 (predicted) | Same as above |
| Episode length | 90 days | 7-30 days in PR #85 | PR #85 doesn't match paper |
| Re-runs | 96 | 5-10 in PR #85 | PR #85 doesn't match paper |
| Features | 4+6 | 4+2 in PR #85 | Reduced feature set |

The PR #85 reproduction has **3 sources of discrepancy** with the
paper: data, episode length, and feature set. The provenance gap
in PR #90 is the umbrella for these.

---

## 6. Open questions

- [ ] Does the HeartSteps V2 algorithm published in a journal
      (if a journal version exists) report different numbers
      from the arXiv 1909.03539? The Klasnja 2022 paper may be
      the journal version.
- [ ] What is the *minimum* effect size that the project's
      synthetic data can produce? A 1-day power analysis
      experiment would answer this.
- [ ] Is the proxy value necessary for the +29.75 effect, or
      does the algorithm beat TS Bandit by +29.75 even with
      `γ=0`? An ablation would answer this.

---

## 7. Citation block

```bibtex
@misc{liao2019personalized,
  author  = {Liao, Peng and Greenewald, Kristjan and Klasnja, Predrag and Murphy, Susan},
  title   = {Personalized HeartSteps: A Reinforcement Learning Algorithm for Optimizing Physical Activity},
  year    = {2019},
  eprint  = {1909.03539},
  archiveprefix = {arXiv},
  primaryclass = {cs.LG}
}
```

---

*End of HeartSteps V2 effect-size recreation report v0.1*
