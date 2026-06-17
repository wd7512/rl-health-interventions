---
title: "Recreation — Russo & Van Roy 2018 TS regret bounds"
status: "recreation report v0.1"
date: "2026-06-14"
paper: "Russo, D. & Van Roy, B. (2018). Learning to Optimize via Posterior Sampling. Mathematics of Operations Research, 43(4), 1137–1167."
arxiv: "https://arxiv.org/abs/1301.2609"
purpose: "Test whether the project's TS algorithm matches the canonical theoretical regret bound"
related: "PR #85 (HeartSteps V2 reproduction) · PR #86 (statistical analysis plan) · PR #88 (online/offline framing)"
---

# Recreation — Russo & Van Roy 2018 TS regret bounds

> Apply the four-question loop (theoretical limit? link to other work?
> room for improvement? take action) to Russo & Van Roy's Bayesian
> regret bound for Thompson Sampling, in the context of the
> rl-health-interventions project.

---

## 1. What the paper says (in 1 paragraph)

Russo & Van Roy prove a Bayesian regret bound for Thompson Sampling
in stochastic bandits and MDPs that scales as `O(√T)` — specifically
`R(T) ≤ O(√(dT log T))` for linear-reward bandits with `d` features,
under a Gaussian prior. The key technical contribution is the
**information ratio**: the ratio of expected squared regret to the
mutual information gain. They show that TS achieves a near-optimal
information ratio in many settings, which is what produces the
`O(√T)` rate. The bound is **Bayesian** (averaged over a prior on
the reward parameters), not worst-case frequentist.

This is the canonical paper any Nature Methods submission that uses
TS in health will need to cite to justify "principled exploration."

---

## 2. What the paper's headline result is

**Theorem 1 (informal):** For a linear bandit with reward
`E[r | x, a] = x^T θ*` and Gaussian prior `θ* ~ N(0, λI)`, the
Bayesian regret of TS satisfies:

```
E[R(T)] ≤ O(d √(T log T) · √(log(1/δ)))
```

where `δ` is a failure probability and the expectation is over the
prior and the algorithm's internal randomness.

**Key consequence for our project:** The HeartSteps V2 algorithm
(PR #85) uses linear Thompson Sampling with a Gaussian prior on
the reward parameters (the posterior is conjugate). It satisfies
the theorem's preconditions. Therefore the algorithm's expected
regret, *averaged over a plausible prior on the reward
parameters*, scales as `O(√(T))`.

This is a strong, citable, theoretical guarantee.

---

## 3. The four-question loop

### 3.1 Is this at the theoretical limit?

**Lower bound:** The information-theoretic lower bound for
stochastic bandits is `Ω(√(dT))` (Lai & Robbins 1985; for linear
bandits, see Dani, Hayes, Kakade 2008). Russo & Van Roy's TS bound
matches this up to logarithmic factors. **Yes, it is at the
theoretical limit** for Bayesian regret.

**But the limit depends on the prior being correct.** The
`O(√T)` rate is averaged over a prior. If the prior is misspecified
(e.g. wrong noise variance, wrong feature scaling), the bound
can be substantially worse in practice. The HeartSteps V2 paper
addresses this with action-centering (PR #85 §2.2 of the review),
which makes the treatment-effect parameter `β` robust to baseline
misspecification. The Russo-Van Roy bound **does not** explicitly
cover this case.

**Verdict:** Theoretical-limit for the bounded case; the
project's action-centering goes beyond the bounded case. There is
a gap in the literature here that the project could potentially
help close (see §3.4).

### 3.2 How does this link to other papers and this work?

| Paper | Link to Russo-Van Roy |
|---|---|
| Liao et al. 2019 (HeartSteps V2) | Uses TS; relies on the O(√T) bound to justify exploration. The proxy value is an additional layer on top. |
| Chowdhury & Gopalan 2017 (kernel bandits) | Proves a similar bound for kernel TS. Different model class, same rate. |
| Karine 2024 (StepCountJITAI) | Reports RL outperforms TS, but does not cite Russo-Van Roy. A literature gap in the simulator paper. |
| PR #86 (statistical analysis plan) | Currently specifies a fixed sample size N=200, K=30. The O(√T) bound suggests *why* 200 is enough: 200 episodes gives O(√200) ≈ 14 step regret bound. |
| PR #88 (online/offline framing) | The Russo-Van Roy bound is for online learning. If the project reframes as "offline + OPE" (PR #88's recommendation), the bound is replaced by OPE error bounds (Jin, Kallus, Saust 2021 etc). The choice of framing determines which theoretical result backs the paper. |

### 3.3 Is there still room for improvement?

Three open theoretical questions worth flagging:

1. **Misspecified prior.** The Russo-Van Roy bound assumes the
   prior is correct. The HeartSteps V2 paper uses a
   data-constructed prior. What is the regret bound when the
   prior is learned? Recent work (e.g. Bhatt et al. 2022 on
   meta-learning for bandits) addresses this; the
   rl-health-interventions project could cite both.

2. **Heterogeneous participants.** The standard bandit setting
   is a single learner. HeartSteps has 37 participants with
   heterogeneous response patterns. The project uses the same
   prior for all participants, which is a simplification.
   Recent work on clustered/contextual meta-bandits (e.g.
   Hong et al. 2022) could be cited.

3. **Delayed feedback.** The standard TS bound assumes immediate
   reward. The HeartSteps V2 paper explicitly models 3-week
   delayed effects. Russo-Van Roy does not cover this. The
   closest is Jin, Jordan, Pal (2021) on batched bandits, but
   that is a different model (batched observations, not delayed
   reward). The project is contributing to an open theoretical
   question.

### 3.4 Take action

Five concrete actions, in priority order:

1. **Add Russo-Van Roy 2018 to references.bib** with full
   BibTeX. Currently missing. This is a 1-line PR.

2. **Cite Russo-Van Roy in the paper's introduction** when
   motivating Thompson Sampling. The paper currently says
   "Thompson Sampling as the baseline agent" with no
   theoretical justification. The Nature reviewer will ask.

3. **In PR #86 (statistical analysis plan), add a subsection on
   why N=200 is sufficient.** The Russo-Van Roy bound gives
   `O(√T)` — at T=200, the bound is `O(14)`. This is a
   *qualitative* argument, not a power calculation, but it
   bridges theory and the analysis plan.

4. **In PR #88 (online/offline framing), note that the
   Russo-Van Roy bound applies only to the online-RL framing.**
   If the project reframes as simulation-based policy
   evaluation (PR #88's recommendation E), the bound is
   replaced by OPE error bounds. The framing decision
   determines which theoretical result backs the paper.

5. **For the project's paper, consider running a finite-T
   regret simulation** (a small experiment, not a full
   reproduction) that compares TS regret to the
   `O(√(dT log T))` bound with different feature dimensions.
   This is a 1-2 day experiment, not a 6-week paper
   reproduction. It would be a novel contribution: most
   papers report regret curves; few report the *constant
   factor* in front of the `√T` term.

---

## 4. What this recreation validates (be honest)

- The recreation is **conceptual**, not numerical. The paper
  proves a theorem; we cannot "run" a theorem.
- The recreation validates that **the project's TS
  implementation falls within the theorem's preconditions**
  (linear reward model, Gaussian prior, contextual bandit
  setting). It does not validate the implementation itself.
- A real numerical recreation would require running TS on a
  simple linear bandit and measuring the regret curve. This
  is a 1-day experiment that the project should add to M-08
  (Evaluation Framework).

---

## 5. Numbers from the paper that the project should know

| Number | Value | What it tells us |
|---|---|---|
| Regret rate | `O(√(dT log T))` | The headline bound |
| Lower bound (matching) | `Ω(√(dT))` | The bound is tight up to log factors |
| Required preconditions | Linear reward, Gaussian prior, contextual bandit | HeartSteps V2 satisfies all three |
| Does NOT cover | Misspecified prior, heterogeneous participants, delayed reward | All three are present in the project |

---

## 6. Open questions

- [ ] Should the project run a finite-T regret experiment to
      validate the implementation matches the bound? Estimated
      effort: 1-2 days, 1 figure.
- [ ] Does the action-centering in PR #85 invalidate the
      bound? Need to check whether Russo-Van Roy's proof
      extends to action-centered linear models. Estimated
      effort: 1 day literature search.
- [ ] Is there a more recent bound (2020-2026) that covers
      the HeartSteps V2 setting more directly? Estimated
      effort: 1 hour search.

---

## 7. Citation block

```bibtex
@article{russo2018learning,
  author  = {Russo, Daniel and Van Roy, Benjamin},
  title   = {Learning to Optimize via Posterior Sampling},
  journal = {Mathematics of Operations Research},
  volume  = {43},
  number  = {4},
  pages   = {1137--1167},
  year    = {2018},
  doi     = {10.1287/moor.2017.0908}
}
```

---

*End of Russo-Van Roy recreation report v0.1*
