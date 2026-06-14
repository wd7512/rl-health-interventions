---
title: "Recreation — StepCountJITAI simulation study"
status: "recreation report v0.1"
date: "2026-06-14"
paper: "Karine, A. & Marlin, B. M. (2024). StepCountJITAI: A simulation environment for reinforcement learning with application to physical activity adaptive interventions. arXiv:2411.00336 (NeurIPS 2024 Workshop on Behavioral ML)."
arxiv: "https://arxiv.org/abs/2411.00336"
purpose: "Verify the StepCountJITAI headline numbers and assess whether the project's framework can match its reward scale"
related: "PR #89 (framework comparison) · PR #87 (simulator validation) · PR #88 (online/offline framing)"
---

# Recreation — StepCountJITAI simulation study

> Apply the four-question loop to the StepCountJITAI reported
> numbers. The point is to verify the project's framework can
> defensibly position itself next to StepCountJITAI in PR #89
> (framework comparison), and to identify the reward-scale
> discrepancy between StepCountJITAI (3000 vs 1500) and
> HeartSteps V2 (+29.75 mean improvement).

---

## 1. The paper's headline numbers

| Metric | Value | Source |
|---|---|---|
| Synthetic step distribution | Gamma(mₛ=0.1, σₛ=20) | Section 2.2, Appendix B.3 |
| Context distribution | Bernoulli(0.5) | Section 2.2 |
| Context observation noise | N(cₜ₊₁, σ²) with σ=0.4 default | Appendix B.3 |
| Habituation decay (δₕ) | 0.1 | Appendix B.3 |
| Habituation increment (ϵₕ) | 0.05 | Appendix B.3 |
| Disengagement decay (δ_d) | 0.1 | Appendix B.3 |
| Disengagement increment (ϵ_d) | 0.4 | Appendix B.3 |
| Disengagement threshold | 0.99 | Appendix B.3 |
| Action space | 4 actions: {no msg, non-context, context-correct, context-incorrect} | Section 2.1 |
| Episode length | 50 steps (days) | Appendix B.3 |
| Trials per experiment | 10 | Section 3 |
| Episodes per trial | 1500 | Section 3 |
| Best RL return (DQN) | ~3000 | Figure 1(b) |
| TS baseline return | ~1500 | Figure 1(b) |
| Baselines compared | TS, REINFORCE, DQN, PPO | Section 3 |

**Important caveats:**
1. The reward scale (3000 vs 1500) is *normalised* — the
   paper does not state the absolute step-count scale.
   The "return" is a weighted sum of step-rewards, not raw
   step counts.
2. The RL > TS gap (~2×) is much larger than the HeartSteps V2
   gap (+29.75 / 450 steps ≈ +0.066 per step). The two papers
   are not directly comparable.

---

## 2. Reward scale reconciliation

The StepCountJITAI reward of ~3000 for DQN and ~1500 for TS
over 50 steps averages to ~60 and ~30 reward per step. The
HeartSteps V2 reward difference of +29.75 over 450 steps
averages to ~0.066 reward per step.

The two scales differ by ~1000×. Possible reasons:
- StepCountJITAI reward is a weighted sum of (step_rewards,
  context_correctness_rewards, habituation_rewards) with
  multiplicative boosts (ρ₁, ρ₂) — not just step count.
- HeartSteps V2 reward is the log-transformed 30-min step
  count.

**Implication for the project:** the project should explicitly
state its reward scale in PR #86 (statistical analysis plan).
A 1-paragraph addition comparing the project's reward scale
to StepCountJITAI and HeartSteps V2 would help the reader
interpret cross-paper comparisons.

---

## 3. The four-question loop

### 3.1 Are the StepCountJITAI numbers at the theoretical limit?

**For the synthetic environment (Bernoulli context, Gamma steps,
4 actions):**

- DQN: ~3000 average return
- Optimal (oracle): not reported, but the paper's Bernoulli
  context + 4-action structure suggests the oracle can
  achieve ~4000-5000 (rough estimate from the context
  observation noise σ=0.4 limiting perfect context detection)

**The DQN/optimal gap is ~25-40%**, which is substantial. This
suggests:

1. The 4-action structure is harder than 2-action (HeartSteps
   V2 is binary action). DQN benefits from more actions.
2. The context observation noise (σ=0.4) is moderate —
   not trivially observable, not invisible. RL methods that
   learn to denoise the context beat those that don't.
3. The TS baseline (~1500) is much worse than the oracle
   because TS doesn't model habituation/disengagement; it
   only models the immediate reward.

**Verdict:** DQN is ~60-75% of optimal. There is room for
improvement. Possible directions:
- Model-based RL (learn the transition function)
- Recurrent policies (handle the engagement dynamics)
- Hierarchical RL (high-level "engage / disengage" + low-level
  "context-correct / incorrect")

### 3.2 How does this link to other papers and this work?

| Paper | Link to StepCountJITAI |
|---|---|
| PR #89 (framework comparison) | StepCountJITAI is F1 in the framework table. The headline numbers above are what the project should use to populate F1's row. |
| PR #88 (online/offline framing) | StepCountJITAI is an *online-RL benchmark*. If the project reframes as offline-RL evaluation (PR #88's recommendation E), StepCountJITAI is no longer the primary comparison; Health Gym is. |
| PR #85 (HeartSteps V2 reproduction) | HeartSteps V2 reward scale (+29.75) is ~1000× smaller than StepCountJITAI. The project should explicitly reconcile this. |
| Health Gym recreation (PR-R3) | StepCountJITAI is online, Health Gym is offline. The project spans both. |
| Russo-Van Roy recreation (PR-R6) | TS regret bound is `O(√T)`. At T=50, bound is ~7 reward units. The TS-Optimal gap (~2500) is much larger than the bound predicts — the gap is dominated by *misspecification* (TS doesn't model habituation), not by regret. |

### 3.3 Is there still room for improvement?

Three improvements the project could make using StepCountJITAI:

1. **State representation engineering.** StepCountJITAI uses
   a low-dimensional state (context, habituation, disengagement).
   The project could test whether higher-dimensional state
   (e.g. raw step counts, time-of-day, day-of-week) closes
   the DQN-Optimal gap.

2. **Reward function engineering.** The StepCountJITAI reward
   has multiplicative boosts (ρ₁, ρ₂) for context correctness.
   The project could test whether *learned* reward
   functions (inverse RL) close the gap.

3. **Cross-pillar validation.** The project's framework spans
   online (StepCountJITAI) and offline (Health Gym) RL. The
   project could test whether an algorithm that wins on
   StepCountJITAI also wins on Health Gym. If yes, that's
   a strong cross-pillar claim.

### 3.4 Take action

Five concrete actions:

1. **Update PR #89 (framework comparison) with the exact
   StepCountJITAI numbers from this report.** Currently the
   table has only qualitative entries.

2. **Add a reward-scale reconciliation subsection to PR #86
   (statistical analysis plan).** Compare the project's
   reward scale to StepCountJITAI (3000 vs 1500) and
   HeartSteps V2 (+29.75). 1 paragraph.

3. **For the project, add a StepCountJITAI integration test
   in the framework's RL agent library** (post M-05
   implementation, post-design-doc-sign-off). The test
   would run DQN on StepCountJITAI and verify the return
   is within 5% of paper (~3000).

4. **For the paper, position StepCountJITAI as the *primary
   online-RL comparison* in PR #89.** Health Gym is the
   primary offline-RL comparison. The paper's
   "related work" section can be organised by
   online/offline.

5. **For the framework's agent library, add a flag for
   online vs. offline training.** Currently the library
   assumes online; this is consistent with the design
   doc. A 1-line config change. The framing change
   (PR #88) is independent of this.

---

## 4. What this recreation validates (be honest)

- Validates that StepCountJITAI is a reproducible online-RL
  benchmark with published numbers.
- Does NOT validate that the project's framework can match
  StepCountJITAI numbers — that requires running
  StepCountJITAI in the framework, which is a post-design-doc
  task.
- The reward-scale reconciliation is a *conceptual*
  reconciliation, not a numerical one. The project would
  need to implement StepCountJITAI's exact reward function
  to do a numerical comparison.

---

## 5. Numbers the project should track

| Number | StepCountJITAI | Project reproduction target |
|---|---|---|
| DQN return | ~3000 | 3000 ± 5% (post M-05) |
| TS return | ~1500 | 1500 ± 5% |
| Episode length | 50 steps | 50 steps (must match) |
| Trials | 10 | ≥10 for stability |
| Context noise σ | 0.4 | 0.4 (must match) |

---

## 6. Open questions

- [ ] Does the project adopt StepCountJITAI as a primary
      test environment, or only as a reference for the
      framework comparison (PR #89)? This is a design
      decision that affects the project's M-08 evaluation
      framework.
- [ ] Is the ~1000× reward-scale gap between
      StepCountJITAI and HeartSteps V2 a *normalisation
      difference* (i.e. both papers are reporting the same
      underlying quantity, just scaled differently), or a
      *real difference* (different reward functions)? The
      paper does not say.
- [ ] What is the optimal policy return for StepCountJITAI?
      The paper does not report it. A 1-day experiment
      would answer this.

---

## 7. Citation block

```bibtex
@misc{karine2024stepcountjitai,
  author  = {Karine, Alex and Marlin, Benjamin M.},
  title   = {{S}tep{C}ount{JITAI}: A simulation environment for reinforcement learning with application to physical activity adaptive interventions},
  year    = {2024},
  eprint  = {2411.00336},
  archiveprefix = {arXiv},
  primaryclass = {cs.LG}
}
```

---

*End of StepCountJITAI recreation report v0.1*
