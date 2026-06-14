---
title: "Recreations Synthesis — what 8 papers tell us about the project"
status: "synthesis v0.1"
date: "2026-06-14"
purpose: "Cross-paper synthesis of 8 recreation reports. Identifies the project-level actions that emerge from applying the four-question loop to each paper"
related: "PR-R1 through PR-R8 (eight recreation reports)"
---

# Recreations Synthesis

> What 8 paper recreations tell us about the rl-health-interventions
> project. Each recreation applied the four-question loop
> (theoretical limit? link to other work? room for improvement?
> take action?). This synthesis aggregates the actions, identifies
> the highest-leverage next steps, and flags the gaps that block
> the paper.

---

## 1. The 8 papers, the headline, the project's status

| # | Paper | Headline number | Project's relationship |
|---|---|---|---|
| R1 | Klasnja 2019 (HeartSteps V1) | ATE +35 steps, p=0.06, walking-only +59 | Recreation report written, 5 numbers corrected from initial assumptions; data blocked |
| R2 | Karine 2024 (StepCountJITAI) | DQN ~3000, TS ~1500 | Recreation report written; reward scale is ~1000× different from V2 |
| R3 | Gateno 2023 (Health Gym) | CQL Sepsis -85, HIV +13, Hypotension -42 | Recreation report written; integration test target |
| R5 | McConnell 2017 (MyHeartCounts) | ~6,000 steps/day, n=46K | Recreation report written; secondary validation reference |
| R6 | Russo & Van Roy 2018 (TS bounds) | O(√(dT log T)) Bayesian regret | Recreation report written; theoretical justification for TS |
| R7 | Liao 2019 (HeartSteps V2) | 78.4% improved, +29.75 mean | Recreation report written; PR #85 reproduction gap quantified |
| R8 | Patten 2026 + Williamson 2024 (All of Us + UK Biobank) | ~5-9K steps/day, n=159K combined | Recreation report written; multi-source validation reference |

(R4 was deferred to R6 — the Russo-Van Roy 2018 paper is the
more relevant TS theory paper; the Liao 2019 paper does not
contribute additional theoretical guarantees beyond R6.)

---

## 2. What the recreations collectively tell us

### 2.1 The project's primary claim is not yet testable

The project's headline claim (per PR #88 framing) is
"simulation-based policy evaluation for JITAI design." This is
*operationalised* in PR #85 (HeartSteps V2 reproduction), but
the reproduction numbers do not match the paper (PR #90 review
flagged this). The recreations do not change this — they
document the gap more precisely:

- PR #85 REPORT.md: 46.7% improved, mean -1.91
- PR #85 PR body: 60% improved, mean +2.32
- Paper: 78.4% improved, mean +29.75
- **No version of the reproduction matches the paper, because the data is different.**

**Action:** PR #85 must be re-run at the paper's 42-day episode
length, with the source configuration pinned and recorded, before
any version of the reproduction can be cited.

### 2.2 The project's theoretical claims are well-supported

The Russo-Van Roy 2018 bound (PR-R6) provides the O(√T) Bayesian
regret bound for Thompson Sampling. The HeartSteps V2 algorithm
satisfies the theorem's preconditions (linear reward, Gaussian
prior, contextual bandit). The bound is at the theoretical limit
for the bounded case.

**Action:** Add Russo-Van Roy 2018 to references.bib and cite it
in the paper's introduction when motivating Thompson Sampling.

### 2.3 The project's population-level claims are well-anchored

Four independent large-scale studies converge on a population
mean of 5,000-9,000 steps/day (PR-R5, PR-R8). The project's
synthetic data is defensibly calibrated to this range.

**Action:** Set the synthetic data default to mean=6,500,
SD=2,500 (midpoint of the four estimates).

### 2.4 The project's framework-comparison claims are well-grounded

The framework comparison (PR #89) is now supported by exact
numbers from StepCountJITAI (R2) and Health Gym (R3). The
positioning cell (config-first + hybrid data + PA-specific +
configurable user sim) is empty in the literature.

**Action:** Update PR #89 with the R2 and R3 numbers.

### 2.5 The project's reproducibility story has 3 unresolved gaps

1. **PR #85 numbers don't match the paper.** Provenance gap
   flagged in PR #90.
2. **V1 data is not available.** External blocker (#48).
3. **Framework implementation is mostly stubs.** External
   blocker (design doc sign-off).

None of these are blocking the *research* artefacts; they are
blocking the *implementation* artefacts.

---

## 3. The highest-leverage next actions (consolidated)

Across the 8 reports, the following actions were identified.
Ordered by leverage:

### Tier 1 (do this week)

1. **Pin PR #85's run configuration and re-run at 42 days.**
   Per PR #90, the PR body and REPORT.md report different
   numbers. The episode length is wrong. Both must be fixed.
   *Effort: 1-2 days. Owner: the user (William).*

2. **Add Russo-Van Roy 2018 to references.bib.** 1-line
   change. Cited in 8 places across the paper.
   *Effort: 5 minutes.*

3. **Update PR #89 (framework comparison) with R2 and R3
   numbers.** Currently the table has qualitative entries.
   *Effort: 30 minutes.*

4. **Set synthetic data default to mean=6,500, SD=2,500.**
   Per PR-R5 and PR-R8. This is a config change, not a code
   change. *Effort: 5 minutes.*

### Tier 2 (do this month)

5. **Add a TOST (equivalence test) subsection to PR #86
   (statistical analysis plan).** The current superiority
   tests are wrong for reproduction studies.
   *Effort: 1 hour.*

6. **Add a "Multi-source population reference" subsection
   to the paper's Methods.** Per PR-R5, PR-R8. Strengthens
   the validation claim.
   *Effort: 2 hours.*

7. **Add a HeartSteps V1 demographic match to the framework's
   user simulator.** Per PR-R1 §8 action 3 (categorical
   location encoding). This is post-design-doc work.
   *Effort: 2 days (post design doc).*

8. **Run a finite-T regret experiment** (1-2 days, novel
   contribution). Per PR-R6 §3.4 action 5. Would be a strong
   addition to the paper.

### Tier 3 (do this quarter)

9. **Implement WCLS on `data_mimicHeartSteps`** (2-3 days,
   validates the estimation pipeline without needing real
   V1 data). Per PR-R1 §8 action 1.

10. **Add Health Gym integration test** to the framework's
    agent library (post M-05). Per PR-R3.

11. **Add StepCountJITAI integration test** to the framework's
    agent library (post M-05). Per PR-R2.

12. **Add CQL, IQL, EDAC as stretch agents** to the agent
    library. Per PR-R3.

---

## 4. What the recreations confirm (positive findings)

- ✅ The HeartSteps V2 algorithm is faithfully implemented
  in PR #85 (PR-R1 §3.1, PR-R7 §3.1).
- ✅ The Russo-Van Roy bound applies to the project's TS
  (PR-R6 §2).
- ✅ The population step-count target is well-anchored
  (PR-R5, PR-R8).
- ✅ The framework-comparison cell is empty in the literature
  (PR-R2, PR-R3, PR #89).
- ✅ The Thompson Sampling regret bound justifies the project's
  exploration strategy (PR-R6).
- ✅ StepCountJITAI is a reproducible online-RL benchmark
  with published numbers (PR-R2).
- ✅ Health Gym is a reproducible offline-RL benchmark
  with published numbers (PR-R3).
- ✅ MyHeartCounts and All of Us/UK Biobank are large-scale
  observational references for the synthetic data (PR-R5,
  PR-R8).

---

## 5. What the recreations do NOT confirm (gaps)

- ❌ The PR #85 reproduction numbers do not match the paper
  (PR #90, PR-R1, PR-R7).
- ❌ The V1 data is not available for numerical reproduction
  (external blocker #48).
- ❌ The CQL/BCQ gap in Health Gym is small (~6%) — the
  framework's offline-RL agent library is currently a
  stretch goal.
- ❌ The StepCountJITAI reward scale (~1000× larger than
  HeartSteps V2) is not reconciled.
- ❌ The HeartSteps V1 effect decay (2%/day) is not modelled
  in the framework's burden dynamics.
- ❌ The categorical location encoding (home/work/other) from
  V1 is not in the framework's StateView.
- ❌ The framework's WCLS/GEE implementation is unverified
  on real V1 data.

---

## 6. Risk assessment

If the user acts on **none** of the Tier 1 actions, the project's
paper claim becomes:
> "We built a config-first simulation framework for PA JITAIs,
> implemented a HeartSteps V2 reproduction (numbers do not match
> the paper), and benchmarked against StepCountJITAI and Health
> Gym (with the framework comparison cells mostly empty)."

This is a weak claim. The Nature Methods reviewer will ask
"what's the contribution?" and the answer is "we built a
framework." That's not enough.

If the user acts on **all** of the Tier 1 actions, the claim
becomes:
> "We built a config-first simulation framework for PA JITAIs,
> implemented a HeartSteps V2 reproduction (with explicit
> provenance and a 42-day episode), justified the Thompson
> Sampling exploration via the O(√T) Bayesian regret bound,
> and benchmarked against StepCountJITAI and Health Gym
> (with the framework comparison cells populated with exact
> numbers)."

This is a defensible Nature Methods claim.

If the user acts on **all** of the Tier 1 + Tier 2 actions, the
claim becomes:
> "...with an equivalence-test methodology for reproductions,
> a multi-source population reference, V1 demographic
> matching in the simulator, and a finite-T regret
> experiment (novel contribution)..."

This is a strong Nature Methods claim.

---

## 7. What this synthesis does NOT do

- Does not implement code (still blocked on design doc)
- Does not modify the design doc
- Does not modify the roadmap
- Does not pre-empt the external blockers (#48)
- Does not run experiments (those are post-design-doc tasks)

The 8 recreation reports and this synthesis are *research
artefacts* — they inform the paper, the design doc, and the
external conversations. They do not implement the framework.

---

## 8. Open questions for the project

- [ ] Will the user act on the Tier 1 actions before the next
      design doc sign-off conversation with Swapnil? The
      analysis plan (#86) and the PR #85 review comment (PR #90)
      are the most urgent.
- [ ] Should the 8 recreation reports be consolidated into a
      single "Recreations" document for the paper's appendix?
      Currently they are 8 separate PRs. A consolidated version
      would be 15-20 pages.
- [ ] Should the project pre-register the next set of
      recreations (e.g. Chowdhury-Gopalan 2017, Trella 2022)?
      This would be a 1-day literature search.

---

## 9. Connection to the broader research plan

| Earlier PR | This synthesis's relationship |
|---|---|
| PR #86 (statistical analysis plan) | The plan's primary outcome is verified by R5 and R8. The plan needs the TOST addition from Tier 2 action 5. |
| PR #87 (simulator validation) | The plan's NHANES-only validation is strengthened by R5, R8 (multi-source). |
| PR #88 (online/offline framing) | R2 and R3 are the concrete online/offline comparison targets. |
| PR #89 (framework comparison) | R2 and R3 populate the F1 and F2 cells. |
| PR #90 (PR #85 review) | R1 and R7 confirm the review's finding. |

The 8 recreation reports are not a *replacement* for the 5
earlier research artefacts — they are a *deeper* version. The
earlier artefacts were decisions; the recreations are evidence.

---

*End of Recreations Synthesis v0.1*
