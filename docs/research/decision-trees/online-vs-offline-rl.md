# Decision Tree: Online vs Offline RL Framing for `rl-health-interventions`

**Author:** William Dennis
**Date:** 2026-06-14
**Status:** Draft for review
**Context:** This artefact resolves a tension between the project's stated online-RL framing
(initial_design.tex §1) and supporting evidence for an offline-RL component
(PR #85 HeartSteps V2 reproduction, agent library stretch goals, audit finding).

---

## 1. The Tension

The project description in `initial_design.tex` says:

- "RL-in-the-loop" (title, §1)
- "Thompson Sampling as the baseline agent" (§1, §4)
- "deploy the JITAI, learn from it" (implied by "RL-in-the-loop")
- "Phase 2 targets HeartSteps V1/V2 data" (§5)

This reads as an **online RL** framing: the agent interacts with users, collects
data, and adapts in real time. Thompson Sampling is the canonical online learning
algorithm for contextual bandits.

However, several signals point in a different direction:

| Signal | Source | What it says |
|--------|--------|--------------|
| Audit finding | AUDIT_MASTER.md | "Offline RL for Phase 2 not motivated despite limited MRT data" |
| PR #85 | `rl-health-interventions-repro/paper_reproduction/heartsteps/agent.py` | Literal offline-RL reproduction of Liao et al. 2019 (arXiv:1909.03539) — the HeartSteps V2 TS is being **replayed on logged data** |
| Agent library stretch goals | subphase-1d-agent-library.md | CQL, IQL listed as future agents — both are offline-RL algorithms |
| Phase 2 description | initial_design.tex §6 | "Offline RL (CQL, IQL)" listed as stretch goals |
| Data strategy | initial_design.tex §5 | Phase 1: synthetic data only. Phase 2: logged HeartSteps data |
| No MRT fielding plan | Entire design doc | No IRB protocol, no real deployment timeline, no online learning safety protocol |

**The unresolved question:** Is `rl-health-interventions` an online RL framework,
an offline RL framework, or something else? What does the paper actually claim?

---

## 2. The Decision Tree

### Candidate Framings

```
┌─────────────────────────────────────────────────────┐
│           What is the paper's RL framing?           │
├──────────┬──────────┬──────────┬──────────┬─────────┤
│          │          │          │          │         │
│  A       │  B       │  C       │  D       │  E      │
│ Pure     │ Pure     │ Hybrid   │ Sim-to-  │ Policy  │
│ Online   │ Offline  │ Offline  │ Real     │ Eval.   │
│ RL       │ RL       │ ─→ Online│          │ (rec'd) │
└──────────┴──────────┴──────────┴──────────┴─────────┘
```

---

### Framing A: Pure Online RL

**Description.** The framework enables online RL in JITAIs. Agents deploy in an
MRT-style setting, collect experience from users, and update policies in real
time. The config-driven simulation is a dry-run before fielding.

**Requirements.**

| Category | Requirement | Feasibility |
|----------|------------|-------------|
| Data | None pre-collected; agent generates its own data | ✅ Synthetic in Phase 1 |
| MRT access | Required for Phase 2; real deployment | ❌ No IRB, no deployment plan |
| Regulatory | IRB approval, safety monitoring board, adverse event protocol | ❌ None documented |
| Safety | Real-time safety constraints, intervention limits, harm detection | ⚠️ M-10 planned but not implemented |
| Compute | Moderate; agent learns per-user online | ✅ |

**Claims it supports.**

- "We provide a simulation environment for testing online RL policies before
  deploying them in MRTs."
- "Thompson Sampling outperforms fixed baselines in an online learning setting."

**Failure modes.**

1. **No real deployment.** Phase 2 targets HeartSteps *logged data*, not a
   new MRT. The paper would claim online RL but never deploy online — a
   fundamental validity gap.
2. **Safety without oversight.** Online RL in health requires a DMC (Data
   Monitoring Committee) and an IRB-approved protocol. Neither exists.
3. **Simulation ≠ reality.** Synthetic user responses do not capture
   real-world non-stationarity, habituation, or side effects. Any online-RL
   claim based on synthetic evaluation is inherently weak.
4. **Liao et al. 2019 is offline here.** PR #85 reproduces the HeartSteps V2
   TS from **logged data** (off-policy evaluation), not from an online
   interaction loop. Calling this "online RL" contradicts what the code
   actually does.

**Primary audience.** ML researchers who build RL-for-health systems. They will
ask: "Where is the real MRT? If you only evaluated in simulation, what's new?"

**Verdict.** Unsupported. The project has no path to real online deployment
within its scope. The audit's Phase 2 description explicitly lists offline-RL
stretch goals, not online deployment.

---

### Framing B: Pure Offline RL

**Description.** The framework is a toolkit for offline RL on pre-collected
health intervention log data. Researchers load logged MRT data (e.g., HeartSteps
V1/V2), train policies with offline-RL algorithms (CQL, IQL, or fitted Q-iteration),
and evaluate them via off-policy evaluation (OPE). The config-driven simulation
is a sandbox for developing OPE estimators.

**Requirements.**

| Category | Requirement | Feasibility |
|----------|------------|-------------|
| Data | Pre-collected MRT logs with actions, rewards, context | ⚠️ HeartSteps requires data-sharing agreement (weeks–months) |
| MRT access | None (logs are static) | ✅ |
| Regulatory | Data use agreements, de-identification | ⚠️ Not documented |
| Compute | Higher than online (fitted Q-iteration, ensemble OPE) | ✅ |
| Off-policy evaluation | Need IS, WIS, FQE, or similar | ⚠️ Not implemented |

**Claims it supports.**

- "We provide a configurable framework for offline RL evaluation on mobile
  health intervention data."
- "CQL/IQL on HeartSteps V2 data recovers or improves upon the original
  TS policy."

**Failure modes.**

1. **No real offline-RL evaluation yet.** Phase 1 uses synthetic data with
   rule-based transitions. Offline RL on synthetic data from a known generative
   process is trivial — the real challenge is offline RL on logged data with
   confounding and distribution shift.
2. **Thompson Sampling is not an offline-RL algorithm.** TS is an online
   bandit algorithm. Using it as the baseline agent in an offline-RL paper
   requires justification (e.g., "we evaluate TS as an off-policy baseline").
3. **Scope mismatch.** The design doc's headline contribution is "a configurable
   simulation framework for JITAIs" — not "an offline RL toolkit." The paper
   would need a major rewrite.
4. **Lack of OPE.** No importance sampling, no FQE, no WIS in scope. Offline RL
   without OPE cannot produce reliable policy comparisons.

**Primary audience.** ML researchers working on offline RL for healthcare. They
will ask: "Which offline-RL algorithms do you support? What is your OPE
methodology? Do you have real logged data?"

**Verdict.** Premature. The framework could support this in Phase 2, but Phase 1
synthetic data does not exercise the offline-RL challenge (distribution shift,
partial coverage). The project needs to decide: is the contribution the
*framework design* (config-first) or the *offline-RL results* (algorithm
performance)?

---

### Framing C: Hybrid (Offline Pretrain → Online Fine-Tune)

**Description.** Agents are pretrained on logged data (Phase 2: HeartSteps logs)
and then fine-tuned online in an MRT. This is the emerging "batch-to-online"
paradigm (e.g., CQL + online fine-tuning, Cal-QL).

**Requirements.**

| Category | Requirement | Feasibility |
|----------|------------|-------------|
| Data | Pre-collected logs + online MRT access | ❌ Both needed, neither confirmed |
| MRT access | Required for fine-tuning | ❌ No IRB, no deployment plan |
| Regulatory | Dual: data use agreement + IRB for online | ❌ |
| Algorithms | Need offline RL (CQL/IQL) + online RL fine-tuning | ❌ Neither implemented |
| Compute | Very high (offline training + online deployment) | ⚠️ |

**Claims it supports.**

- "We demonstrate that offline-pretrained policies can be safely fine-tuned in
  an MRT, reducing the number of real-world interactions needed."

**Failure modes.**

1. **Everything from A plus everything from B.** Requires both logged data
   *and* MRT access. The project has neither confirmed.
2. **No published hybrid benchmark for PA interventions.** Would be novel but
   unvalidated.
3. **Safety constraints even more important.** Online fine-tuning of an
   imperfect offline policy could harm users.

**Primary audience.** Researchers at the intersection of offline and online RL.
Niche. Most health researchers do not have both logged data and MRT access.

**Verdict.** Far future. Not viable for the current paper.

---

### Framing D: Sim-to-Real

**Description.** The agent trains entirely in simulation (Phase 1 synthetic data)
and is then deployed in a real MRT zero-shot or with minimal adaptation. Value
prop is the high-fidelity simulator that makes zero-shot transfer possible.

**Requirements.**

| Category | Requirement | Feasibility |
|----------|------------|-------------|
| Data | Real deployment data to validate simulation fidelity | ❌ |
| MRT access | Required for "real" half | ❌ |
| Regulatory | Same as online RL | ❌ |
| Simulation fidelity | Must be validated against real user behaviour | ❌ Synthetic data only |

**Claims it supports.**

- "We build a high-fidelity JITAI simulator that enables zero-shot policy
  transfer to real users."

**Failure modes.**

1. **Sim-to-real is hard even in robotics with millions of trajectories.**
   For health, where individual variation is high, zero-shot sim-to-real is
   implausible.
2. **Simulation fidelity not validated.** Phase 1 uses rule-based archetypes
   with no real-data calibration.
3. **Unusual for health.** No published sim-to-real health intervention paper
   exists as a template.

**Primary audience.** Sim-to-real researchers (mostly in robotics). Not relevant
for a Nature Methods health audience.

**Verdict.** Overclaim. The synthetic data is too simplistic for a sim-to-real
claim.

---

### Framing E (Recommended): Simulation-Based Policy Evaluation for JITAI Design

**Description.** The framework is a **configurable simulation environment for
off-policy evaluation** of JITAI intervention policies. It does not deploy
agents online. Instead, it allows researchers to:

1. Specify an MDP (state space, action space, reward function, transition model)
   entirely through YAML configs.
2. Evaluate candidate policies (TS, random, rule-based, DQN, PPO) against
   baselines on **either** synthetic user cohorts (Phase 1) **or** logged MRT
   data (Phase 2, via off-policy evaluation).
3. Generate reproducible, config-driven comparison tables without bespoke
   engineering.

The core contribution is the **config-first architecture** that standardises
JITAI policy evaluation across datasets and algorithms. "RL-in-the-loop" in
the design doc is reinterpreted as "RL algorithms evaluated *in the loop* of
the simulation pipeline," not "RL deployed in a real MRT."

**Requirements.**

| Category | Requirement | Feasibility |
|----------|------------|-------------|
| Data | Synthetic (Phase 1) or logged MRT data (Phase 2) | ✅ Phase 1: synthetic data parameterised from 7 open datasets (NHANES, MMASH, etc.) |
| MRT access | None required | ✅ |
| Regulatory | Data use agreements for Phase 2 (HeartSteps) | ⚠️ Acceptable risk |
| Safety | Constraint testing in simulation (M-10) | ⚠️ Planned |
| Off-policy evaluation | IS/WIS or FQE for Phase 2 analysis | ⚠️ Needs implementation |
| Compute | Moderate; simulation + policy evaluation | ✅ |

**Claims it supports.**

- "We present a configurable simulation framework for standardised,
  reproducible off-policy evaluation of JITAI intervention strategies,
  supporting synthetic data (Phase 1) and logged MRT data (Phase 2)."
- "Config-first design enables cross-dataset, cross-policy comparison without
  bespoke engineering."
- "Thompson Sampling evaluated in simulation replicates HeartSteps V2 findings,
  validating the framework's ability to recover known treatment effects."

**Failure modes.**

1. **"So it's just a simulator?"** Risk of appearing incremental. Mitigation:
   the config-first design *is* the novelty. Frame as "a reproducibility
   infrastructure for JITAI research," not a new algorithm.
2. **Off-policy evaluation on synthetic data is trivial.** Synthetic data from
   a known generative process has no confounding. Mitigation: Phase 1 evaluates
   policies via on-policy simulation (train TS in the loop); Phase 2 introduces
   real off-policy evaluation on logged data.
3. **Thompson Sampling is not normally evaluated off-policy.** TS is an online
   algorithm. In Phase 1, it *is* evaluated on-policy (it interacts with the
   simulator). In Phase 2, it is evaluated off-policy from logs. The paper
   should clarify this distinction explicitly.
4. **Nature Methods may expect a clinical finding.** A framework paper may need
   a demonstration result (e.g., "TS outperforms random by 15% in simulation").
   This is achievable.

**Primary audience.** Mixed:
- **Clinical trialists:** "How do I compare intervention strategies before
  running an expensive MRT?" — the framework is a design tool.
- **ML researchers:** "How do I test a new RL algorithm on health data?" — the
  framework standardises the comparison pipeline.
- **Health behaviour researchers:** "Can I see what effect different reward
  formulations have?" — config-driven experimentation.

---

## 3. Which Framing Fits the Project As It Exists?

| Criterion | A: Online | B: Offline | C: Hybrid | D: Sim-to-Real | E: Policy Eval. |
|-----------|-----------|------------|-----------|----------------|-----------------|
| Design doc alignment | ⚠️ Stated but contradicted | ❌ Not stated | ❌ | ❌ | ✅ Closest read |
| PR #85 consistency | ❌ PR #85 is offline replay | ✅ Explicitly offline | ⚠️ Half | ❌ | ✅ Both modes |
| Phase 1 (synthetic) | ✅ Train on synthetic | ⚠️ Offline RL on syn. is trivial | ❌ | ✅ | ✅ On-policy sim. |
| Phase 2 (logged data) | ❌ No real MRT | ✅ Offline RL on logs | ❌ Needs both | ❌ | ✅ Off-policy eval. |
| Audit finding resolution | ❌ Does not motivate offline | ⚠️ Overcorrects | ❌ | ❌ | ✅ Resolves cleanly |
| Stretch goals (CQL, IQL) | ❌ Irrelevant | ✅ Natural fit | ⚠️ Partial | ❌ | ✅ Natural fit |
| Safety constraints (M-10) | ⚠️ Required but won't be real-world tested | ✅ Simulation only | ❌ | ❌ | ✅ Simulation only |
| Regulatory burden | High | Medium | High | High | **Lowest** |
| Path to Nature Methods | Weak (no deployment) | Medium (needs OPE) | Very weak | Very weak | **Strongest** |

**Recommendation: Framing E — Simulation-Based Policy Evaluation for JITAI
Design** (a.k.a. "configurable off-policy evaluation framework").

---

## 4. What Does the Paper's Headline Claim Become?

### Under each framing, the headline changes:

| Framing | Headline / Abstract Claim |
|---------|--------------------------|
| A: Online | "A simulation framework for online RL in JITAIs, enabling safe policy learning before real-world deployment." |
| B: Offline | "A configurable offline RL toolkit for mobile health intervention data, supporting policy evaluation from logged MRTs." |
| C: Hybrid | "Offline-to-online RL for JITAIs: pretraining policies on logged data and fine-tuning in micro-randomized trials." |
| D: Sim-to-Real | "High-fidelity JITAI simulation enables zero-shot policy transfer from synthetic users to real deployments." |
| **E: Policy Eval.** | **"A configurable simulation framework for reproducible off-policy evaluation of JITAI intervention strategies across datasets and algorithms."** |

### Under the recommended framing (E), the paper's contribution is:

> **Primary contribution:** A config-first simulation framework that standardises
> and reproduces JITAI policy evaluation — supporting synthetic data (Phase 1)
> and logged MRT data (Phase 2) through a shared YAML-driven interface.
>
> **Secondary contribution (Phase 1):** Empirical comparison of Thompson
> Sampling, random, fixed, and rule-based policies on synthetic user cohorts
> with 4 behavioural archetypes, demonstrating that the framework recovers
> expected policy ordering.
>
> **Secondary contribution (Phase 2 roadmap):** Off-policy evaluation of
> candidate policies on HeartSteps V2 logged data, validating simulation-based
> findings against real MRT outcomes. CQL/IQL as method demonstrations.

---

## 5. Minimum Change to Commit to Framing E

No edits to `initial_design.tex` or `ROADMAP.md` are required per constraints.
Instead, the following *supplementary* changes establish the framing:

### 5.1 This Decision Tree (Already Done)

The existence of this document (`docs/research/decision-trees/online-vs-offline-rl.md`)
records the decision. Future readers can trace why the framing was chosen.

### 5.2 Add a Three-Sentence Framing Note to the Design Doc

The task forbids edits to `initial_design.tex`. Instead, add a one-page
supplement `docs/design/paper-framing-supplement.tex` that clarifies:

```
\section{Paper Framing Clarification (Supplement to §1)}
The simulation framework evaluates policies via configurable off-policy
evaluation --- agents interact with synthetic users (Phase 1) or are
evaluated on logged data (Phase 2). "RL-in-the-loop" refers to the
simulation loop, not real MRT deployment. Offline RL (CQL, IQL) is
motivated for Phase 2 as the natural evaluation mode for logged MRT data.
```

### 5.3 Agent Library: Rename "Stretch Goals" Section

In `docs/plans/stale/subphase-1d-agent-library.md`, the stretch goals list currently
reads as optional add-ons. Under Framing E, CQL and IQL are not stretch goals —
they are Phase 2 deliverables. The distinction between Phase 1 (on-policy
simulation: TS, DQN, PPO) and Phase 2 (off-policy from logs: CQL, IQL, FQE)
should be made explicit. Since the task forbids edits, note this in the
decision log.

### 5.4 Statistical Analysis Plan: Add Off-Policy Evaluation Section

A `docs/research/archive/statistical-analysis-plan.md` (currently archived)
should be created with:

- **Phase 1 analysis.** On-policy simulation. Compare mean cumulative reward
  across policies using bootstrap CIs (1000 resamples). Effect size: Cohen's d
  between TS and best baseline. Sample size: 100 synthetic users × 90 days,
  guaranteed by generation.
- **Phase 2 analysis.** Off-policy evaluation. Use Weighted Importance Sampling
  (WIS) or Fitted Q-Evaluation (FQE) to estimate target policy value from
  logged HeartSteps data. Report standard errors via bootstrap. Compare TS
  (behavior policy, replayed) vs CQL vs IQL.
- **No sample-size calculation needed for Phase 2** (fixed logged dataset).
  Power analysis is for Phase 1 only.

This is the single most important action: the analysis plan must distinguish
on-policy simulation from off-policy evaluation.

### 5.5 PR #85: Document as Phase 2 Prototype

Add a short README or comment to `rl-health-interventions-repro/` noting that
this reproduction is the Phase 2 offline-RL pipeline prototype. It is not a
Phase 1 deliverable.

---

## 6. Interaction with the Statistical Analysis Plan

Framing E changes the analysis plan in several ways:

### 6.1 Baseline Set

| Framing | Baselines |
|---------|-----------|
| Online RL (A) | Random, fixed, rule-based, plus "deployed TS" as real-world baseline |
| Offline RL (B) | Random, uniform, behaviour-cloning, plus OPE estimates |
| **Policy Eval. (E)** | **Phase 1: Random, fixed, rule-based, TS (on-policy). Phase 2: Behaviour policy (logged TS), CQL, IQL, random (off-policy via OPE)** |

Under E, you need **two baseline sets**: one for synthetic simulation (Phase 1)
and one for logged data (Phase 2). They share common baselines (random,
rule-based) but use different evaluation methodologies.

### 6.2 Sample Size

- **Phase 1 (on-policy simulation):** You control N. Power analysis determines
  minimum users × days to detect a target effect size (e.g., 15% reward
  improvement). This is a standard simulation-based power calculation.
- **Phase 2 (off-policy evaluation):** You do NOT control N. The logged dataset
  is fixed (97 users × 90 days for HeartSteps V2). The analysis uses all
  available data. Precision is determined by the effective sample size of the
  importance weights, not the raw N.

Under Framing A (online), Phase 2 would require a power analysis for a new
MRT — which does not exist.

### 6.3 Primary Outcome

| Framing | Primary Outcome | Secondary |
|---------|----------------|-----------|
| Online (A) | Cumulative reward during MRT | Regret, adherence, safety violations |
| Offline (B) | Off-policy estimated value of target policy | IS weights diagnostics, coverage |
| **Policy Eval. (E)** | **Phase 1: Mean cumulative reward over 90-day episode. Phase 2: WIS-estimated value of target policy ± bootstrap CI** | Regret, adherence, sustained change (Phase 1); OPE diagnostics, effective sample size (Phase 2) |

### 6.4 Statistical Tests

- **Phase 1:** Bootstrap CIs for pairwise comparisons. No multiple-testing
  correction needed if comparisons are pre-specified (<6). Effect sizes with
  Cohen's d.
- **Phase 2:** Bootstrap CIs for WIS estimates. Studentised bootstrap for
  coverage guarantees. Sensitivity analysis: vary OPE estimator (IS vs WIS vs
  FQE) to assess robustness.

### 6.5 Concrete Changes Needed

| Current (implied) | Under Framing E |
|-------------------|-----------------|
| "Compare TS to baselines in simulation" | "Compare TS to baselines in on-policy simulation (Phase 1) and via off-policy evaluation on logged HeartSteps data (Phase 2)" |
| Thompson Sampling as the paper's demo | TS as Phase 1 algorithm; CQL/IQL as Phase 2 algorithms — the paper demonstrates both modes |
| No OPE methodology | WIS + FQE specified in analysis plan |
| No distinction between synthetic and logged analysis | Separate analysis plans with different statistical machinery |

---

## 7. How PR #85 Fits Under Framing E

PR #85 (`rl-health-interventions-repro/paper_reproduction/heartsteps/agent.py`)
reproduces the Liao et al. 2019 HeartSteps V2 Thompson Sampling agent. Under
Framing E, this reproduction serves two purposes:

1. **Phase 2 prototype.** It demonstrates that the framework can load real MRT
   data (HeartSteps V2 logs) and evaluate a TS policy via off-policy replay.
   The Bayesian regression, dosage tracker, and proxy value function are all
   components that would be registered as configurable building blocks.
2. **Validation benchmark.** The reproduced TS policy's estimated value on
   HeartSteps V2 data establishes a reference point. Future agents (CQL, IQL,
   DQN) can be measured against this benchmark using the same config-driven
   pipeline.

The PR #85 agent is **not** an online RL agent — it replays actions from logged
data. This is consistent with Framing E, where Phase 2 uses offline evaluation
on static logs. The audit's concern ("Offline RL for Phase 2 not motivated") is
resolved: PR #85 *is* the offline RL prototype, and Phase 2 *is* the offline
evaluation phase.

---

## 8. Open Questions

1. **What is the exact Phase 1 → Phase 2 boundary?** Does Phase 1 produce
   publishable results (synthetic-only comparison), or is it purely formative?
   Under Framing E, Phase 1 alone is sufficient for a "framework + synthetic
   demonstration" paper (e.g., Nature Methods or Scientific Data).

2. **Is off-policy evaluation in Phase 2 a core contribution or a validation
   exercise?** If it is a core contribution, the paper needs OPE methodology
   development (e.g., a new OPE estimator for MRT data). If validation only,
   standard WIS/FQE suffices.

3. **Does the project need to cite Liao et al. 2019 explicitly?** Currently,
   `references.bib` cites `klasnja2019heartsteps` (the V1 clinical paper) and
   `klasnja2022heartsteps` (the V2 protocol), but not the actual Liao et al.
   arXiv:1909.03539 that PR #85 implements. This gap should be filled.

4. **Who is the target journal?** Nature Methods would expect methodological
   novelty (config-first design). Nature Scientific Data would accept the
   framework as a "resource." A clinical journal would expect a clinical
   finding. The framing choice should align with the target venue.

5. **Should "RL-in-the-loop" be removed from the paper's language?** Under
   Framing E, "RL-in-the-loop" is misleading because the loop is simulation,
   not deployment. Recommended replacement: "RL-in-simulation" or
   "config-driven RL policy evaluation."

---

## 9. Summary: Recommended Framing

```
Framing: E — Simulation-Based Policy Evaluation for JITAI Design
  └─ Phase 1: On-policy simulation with synthetic users
  │     (TS, DQN, PPO interact with rule-based transition model)
  │     └─ Primary outcome: mean cumulative reward
  │     └─ Evaluation: bootstrap CIs vs random/fixed/rule-based baselines
  │
  └─ Phase 2: Off-policy evaluation on logged MRT data
        (CQL, IQL, TS replayed from HeartSteps V2 logs)
        └─ Primary outcome: WIS-estimated policy value
        └─ Evaluation: bootstrap CIs, sensitivity across OPE estimators
```

**Key benefits:**

1. Resolves the audit finding cleanly — Phase 2 is explicitly offline RL.
2. PR #85 slotts in as the Phase 2 prototype.
3. Lowest regulatory burden (no real MRT needed).
4. Config-first architecture is the primary contribution, not algorithm
   performance — easier to defend.
5. Phase 1 alone is publishable (synthetic demonstration), de-risking the
   timeline.
6. Compatible with all existing design decisions (ABC + registry, YAML configs,
   multi-timescale reward, 4 archetypes).
