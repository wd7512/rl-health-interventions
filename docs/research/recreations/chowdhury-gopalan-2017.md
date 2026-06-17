---
title: "Recreation — Chowdhury & Gopalan 2017 kernelized bandits"
status: "recreation report v0.1"
date: "2026-06-14"
paper: "Chowdhury, S. R. & Gopalan, A. (2017). On Kernelized Multi-armed Bandits. ICML 2017."
arxiv: "https://arxiv.org/abs/1707.06347"
purpose: "Verify the kernel-TS regret bound and assess its applicability to HeartSteps V2's linear (and possibly kernel-extended) reward model"
related: "PR-R6 (Russo-Van Roy 2018) · PR #85 (HeartSteps V2 reproduction) · PR #88 (online/offline framing)"
---

# Recreation — Chowdhury & Gopalan 2017 kernelized bandits

> Apply the four-question loop to the Chowdhury-Gopalan 2017
> regret bound for kernelized Thompson Sampling. The point is
> to identify whether the bound applies to the project's
> HeartSteps V2 reproduction (PR #85), which uses a *linear*
> reward model, and whether a kernel extension is a useful
> research direction.

---

## 1. The paper's headline result

**Theorem 1 (informal):** For a kernelized bandit with reward
`E[r | x, a] = f*(x, a)` for some function `f*` in a Reproducing
Kernel Hilbert Space (RKHS) of dimension `d_eff`, the Bayesian
regret of Thompson Sampling satisfies:

```
E[R(T)] ≤ O(√(d_eff T log T))
```

where `d_eff` is the *effective dimension* of the kernel matrix
on the observed data, and the expectation is over the GP prior
and the algorithm's randomness.

**Key contribution:** the bound is *adaptive* — `d_eff` grows
slowly with the data, so the bound is tight even when the
ambient RKHS dimension is infinite (e.g. RBF kernel).

---

## 2. Applicability to the project

### 2.1 HeartSteps V2 (PR #85) uses a *linear* reward model

The HeartSteps V2 algorithm uses:
- 4 treatment-effect features (dosage, app engagement, location,
  step variability)
- 6 baseline features (the 4 above + prior 30-min step count,
  yesterday's total, current temperature)
- A linear reward model: `R_t = g(S_t)^T α₀ + π_t f(S_t)^T α₁ + (A_t - π_t) f(S_t)^T β + ε`

This is *strictly linear*. The Chowdhury-Gopalan bound applies,
with `d_eff = d` (the feature dimension) and kernel = linear
kernel. So:

- For HeartSteps V2, the bound is `O(√(6 T log T))` (using
  `d = 6` for the total parameter count, or `d = 4` for the
  treatment-effect subset).
- The Russo-Van Roy 2018 bound (PR-R6) gives the same rate
  for the linear case. **The two bounds are equivalent for
  HeartSteps V2.**

### 2.2 When would the kernel extension matter?

The kernel extension would matter if the project extends the
reward model from linear to nonlinear. Two natural extensions:

1. **Polynomial features.** Adding `f(s)^2` or `f(s)·g(s)` to
   the feature vector is a polynomial kernel. The kernel bound
   would give `O(√(d² T log T))` — worse than linear but
   captures interactions.

2. **RBF / Gaussian kernel.** A nonlinear reward model. The
   kernel bound would give `O(√(d_eff T log T))` where
   `d_eff` is data-dependent. This is the most general
   extension and would let the project claim "nonlinear
   reward modelling" as a contribution.

**For the current project (HeartSteps V2 reproduction with
linear model), the kernel extension is not needed.** But for
the Phase 2 real-data work, a kernel extension may be a
natural research direction.

---

## 3. The four-question loop

### 3.1 Is the bound at the theoretical limit?

**For kernelized bandits:**

The information-theoretic lower bound is `Ω(√(d_eff T))`
(Srinivas et al. 2010 for GPs; Valko et al. 2013 for linear
TS in kernelized settings). The Chowdhury-Gopalan bound matches
this up to log factors. **Yes, it is at the theoretical limit.**

**For the linear special case (HeartSteps V2):**

The bound reduces to the Russo-Van Roy 2018 rate
`O(√(dT log T))`. The two papers are equivalent in the linear
case; citing one suffices for the paper.

**Verdict:** at the theoretical limit for the kernelized case.
The HeartSteps V2 reproduction doesn't need this paper's
result — Russo-Van Roy 2018 covers it.

### 3.2 How does this link to other papers and this work?

| Paper | Link to Chowdhury-Gopalan |
|---|---|
| Russo-Van Roy 2018 (PR-R6) | Equivalent in the linear case. Chowdhury-Gopalan is the *kernel extension*; Russo-Van Roy is the *linear foundation*. |
| HeartSteps V2 (PR #85) | Linear model; Russo-Van Roy bound suffices. |
| PR #88 (online/offline framing) | If the project reframes as offline-RL evaluation, kernel bounds become less relevant (offline bounds are different — see e.g. Jin, Kallus, Saust 2021). |
| PR #89 (framework comparison) | Adding kernel-TS as a stretch agent would be a contribution; not currently in the agent library. |
| StepCountJITAI recreation (PR-R2) | Karine 2024 uses standard linear TS. A kernel extension would be a natural follow-up. |

### 3.3 Is there still room for improvement?

Three open questions:

1. **Adaptive kernel selection.** The bound assumes a fixed
   kernel. Recent work (e.g. Li & Scarlett 2022) extends to
   *adaptive* kernels, where the kernel bandwidth is learned
   online. The project could test this on HeartSteps V2.

2. **Heteroscedastic noise.** The bound assumes homoscedastic
   noise. The HeartSteps V2 reward (log-step count) is
   approximately homoscedastic, but a heteroscedastic
   extension would be more realistic.

3. **Delayed feedback.** Like Russo-Van Roy, the
   Chowdhury-Gopalan bound assumes immediate reward. The
   HeartSteps V2 reward has a 3-week delayed component
   (via the proxy value). A kernel-TS bound with delayed
   feedback is an open problem.

### 3.4 Take action

Five concrete actions:

1. **Add Chowdhury-Gopalan 2017 to references.bib** if the
   project plans a kernel extension. Otherwise, defer this
   citation.

2. **For the paper, cite Russo-Van Roy 2018 as the primary
   TS theory reference.** The HeartSteps V2 model is linear;
   Russo-Van Roy suffices. Chowdhury-Gopalan is a *future
   work* citation for kernel extensions.

3. **For the project's stretch agent library, add
   kernel-TS as a research direction.** Documented in
   subphase_1d_agent_library.md as a stretch goal.

4. **For the project's Phase 2 real-data work, consider
   testing whether a kernel-TS extension beats linear TS
   on HeartSteps V1 data.** This is a post-V1-data-access
   experiment (external blocker #48).

5. **For the project's M-08 evaluation framework, add a
   "TS variant comparison" subsection.** Compare linear
   TS, kernel-TS, and a contextual bandit baseline. The
   kernel-TS comparison is post-stretch.

---

## 4. What this recreation validates (be honest)

- Validates that the *theoretical* justification for TS in
  HeartSteps V2 is well-established in the literature
  (Russo-Van Roy for linear, Chowdhury-Gopalan for kernel).
- Does NOT validate the *implementation* — that requires
  a regret-curve experiment.
- The paper's primary contribution is a *kernel bound*; for
  the project's linear case, Russo-Van Roy 2018 is the
  sufficient citation.

---

## 5. Numbers from the paper

| Number | Value | What it tells us |
|---|---|---|
| Regret rate | `O(√(d_eff T log T))` | Adaptive bound (d_eff data-dependent) |
| Lower bound (matching) | `Ω(√(d_eff T))` | Tight up to log factors |
| HeartSteps V2 specialisation | `O(√(6 T log T))` with d=6 | Equivalent to Russo-Van Roy |

---

## 6. Open questions

- [ ] Does the project plan a kernel extension of the reward
      model? This is a design decision; current scope is
      linear.
- [ ] Is there a tighter bound for the *binary action* case
      (HeartSteps V2 is binary)? Binary action TS bounds
      may have a smaller constant.
- [ ] Does the project's delay-aware reward (3-week proxy
      value) break the kernel-TS bound? Likely yes; an
      open theoretical question.

---

## 7. Citation block

```bibtex
@inproceedings{chowdhury2017kernelized,
  author  = {Chowdhury, Sayak Ray and Gopalan, Aditya},
  title   = {On Kernelized Multi-armed Bandits},
  booktitle = {ICML},
  year    = {2017},
  eprint  = {1707.06347},
  archiveprefix = {arXiv},
  primaryclass = {stat.ML}
}
```

---

*End of Chowdhury-Gopalan recreation report v0.1*
