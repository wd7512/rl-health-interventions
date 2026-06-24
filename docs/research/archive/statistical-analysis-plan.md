---
title: "Statistical Analysis Plan — rl-health-interventions"
status: "draft v0.1"
date: "2026-06-14"
author: "William Dennis (with research-assistant prep)"
target_journal: "Nature Methods"
pre_registration_intent: "yes — locked before Sprint 3 evaluation work begins"
supersedes: "Open decision in initial_design.tex §5 ('evaluation methodology')"
related: "docs/initial_design.tex · docs/ROADMAP.md M-08 · docs/research/decision-trees/*"
---

> **Archived 2026-06-23.** This document was part of the June-14 research batch
> (PRs #86–#90). It is retained for historical reference but is no longer actively
> maintained. See `docs/research/README.md` for current research artifacts.

# Statistical Analysis Plan

> Pre-registered analysis plan for the simulation experiments that will
> accompany the Nature Methods submission. Locks the metrics, baselines,
> and inferential procedure *before* running the main runs. Anything not
> specified here is exploratory and labelled as such in the paper.

---

## 1. Scope and unit of analysis

**Primary unit of analysis:** the participant (within an MRT-style
simulation). All metrics are reported as participant-level summaries,
then aggregated across the simulated cohort.

**Episode:** one full MRT-length simulation per participant (default
42 days, matching HeartSteps V2 pilot length).

**Run:** one full sweep over N participants with a fixed random seed.
All headline numbers are reported as the distribution over ≥30
re-runs per configuration (different random seeds; same config).

**Why this matters:** every metric below has three levels — *within
participant* (over the episode), *between participant* (across the
cohort for one seed), and *between runs* (across seeds). Standard
errors and CIs are reported at the run level unless stated otherwise.

---

## 2. Primary outcome

| Field | Value |
|---|---|
| Outcome | Cumulative step count over the 42-day episode |
| Source | Daily aggregate from the wearable stream |
| Direction | Higher is better |
| Pre-registered hypothesis (H1) | RL agents (Thompson Sampling, HeartSteps V2 policy) achieve ≥10% higher cumulative steps than the strongest non-RL baseline (fixed-schedule) |
| Pre-registered effect-size threshold | Cohen's d ≥ 0.3 (small-to-medium) on cumulative steps |

**Why cumulative steps and not immediate reward:** The reward signal
in our MDP mixes immediate engagement with a 3-week delayed clinical
measure. Cumulative step count is the closest user-facing analogue
that an external reader can interpret without internal MDP definitions.

**Pre-registration lock:** the threshold (10%, d=0.3) and the metric
definition are frozen at v0.1. Any change after main runs start is a
protocol deviation, logged in §10.

---

## 3. Secondary outcomes

| Outcome | Why it's secondary | Pre-registered? |
|---|---|---|
| Daily step count (per-day series) | Captures trajectory, not just total | Yes — used for time-series plots |
| Adherence rate (proportion of days the user opens the app / engages) | Engagement is a known mediator of intervention efficacy in HeartSteps V1 | Yes |
| Regret (cumulative difference from oracle policy) | Standard RL benchmark | Yes — but **reported alongside, not as primary**, because the oracle is only well-defined in synthetic settings |
| Sustained change (step count in week 6 vs. week 1) | Tests whether the agent just front-loads or builds habit | Yes |
| Burden-adjusted reward (reward − λ·burden) | The internal reward — included for completeness | Yes — but framed as the agent's objective, not a user-facing claim |
| Demographic subgroup gap (steps by age/sex/baseline-activity bin) | Equity requirement for Nature Methods (audit item #14) | **Exploratory** — added 2026-06-14, not pre-registered for the first run |

**Excluded outcomes (with rationale):**

- *Self-reported mood / engagement surveys.* The framework is
  wearable-only by design; surveys are out of scope for Phase 1/2.
- *Composite clinical scores.* No validated composite exists for the
  simulated 42-day window.

---

## 4. Baselines

All baselines share the same MDP, dataset, user simulator, and reward
function. The only thing that varies is the policy.

| Baseline | Description | Why it's necessary |
|---|---|---|
| `random` | Uniform-random action every decision point | Floor — every RL method must beat random in expectation |
| `fixed` | Pre-specified schedule (e.g. push at 17:00 daily, no engagement with state) | The "do nothing smart" comparator — what an industry heuristic achieves |
| `rule_based` | Hand-coded if-then rules (e.g. push if user inactive >2 hours AND time is 9-21) | Captures the type of policy that domain experts actually deploy |
| `contextual_bandit` | Linear-UCB or similar one-step RL with no delayed-reward modelling | Isolates the value of sequential RL over myopic RL |
| `thompson_sampling` (our primary RL baseline) | TS over the linear reward model | Reproduces the HeartSteps V1 baseline |
| `heartsteps_v2` (proposed method, from PR #85) | Liao et al. 2019 with proxy value iteration | The proposed method, from the literature we're reproducing |

**Pre-registration lock:** the baseline set is frozen. Adding a
baseline after the main runs is allowed (labelled "ablation") but
the headline claim is computed against this set.

---

## 5. Sample size and statistical power

**Pre-registered sample size:** N = 200 simulated participants per
condition, K = 30 re-runs (random seeds) per condition.

**Why 200 participants:**
- Mirrors HeartSteps V2's planned enrolment (N≈200) for face validity
- Power calculation (two-sample t, α=0.05, power=0.80) for d=0.3
  requires N≈176 per arm; 200 gives 12% headroom
- 200 is computationally tractable (the simulation runs in minutes
  per participant; 200 × 30 seeds × 7 conditions ≈ 1.5 days on a
  single workstation)

**Why 30 re-runs:** bootstrap stability for between-run CIs.
Empirically, percentile-bootstrap CIs for the median stabilise by
B≈30; we use B=1000 bootstrap replicates within each run for tighter
inference.

**Pre-registration lock:** N=200, K=30, B=1000.

---

## 6. Inferential procedure

### 6.1 Headline comparison

For each (baseline, RL agent) pair, compute the paired difference in
cumulative steps across the N=200 participants within a single seed.
Then aggregate over K=30 seeds.

```
Within-seed:    Δ_{participant, seed} = steps_RL - steps_baseline
Between-seed:   Δ_seed = mean over participants of Δ
Headline:       Δ_hat = mean over seeds of Δ_seed
                SE    = std over seeds of Δ_seed / sqrt(K)
                95% CI = Δ_hat ± 1.96 · SE
                p-value: two-sided paired t-test on Δ_seed values
```

**Multiple comparisons:** 6 baselines × 2 RL agents = 12 headline
comparisons. Apply Benjamini–Hochberg FDR control at q=0.05.

### 6.2 Effect size

Report Cohen's d for paired samples:

```
d = mean(Δ_seed) / std(Δ_seed)
```

Pre-registered thresholds: |d| ≥ 0.2 (small), ≥ 0.5 (medium), ≥ 0.8
(large). The 10% / d=0.3 H1 corresponds to a small-to-medium effect.

### 6.3 Robustness checks (pre-registered)

- **Sensitivity to outlier participants:** re-run with winsorised
  cumulative steps (5th/95th percentile) to check that headline
  results are not driven by 1-2 extreme cases.
- **Sensitivity to seed count:** report headline CIs at K=10, 20, 30
  to show stability.
- **Sensitivity to MDP sign-off outcome:** if Swapnil's MDP review
  changes the transition model parameters by >20%, re-run the full
  matrix under the new MDP. The result is a sensitivity analysis,
  not the primary result.

### 6.4 Exploratory analyses (not pre-registered)

- Demographic subgroup gaps (§3, exploratory row)
- Ablations: with/without delayed reward, with/without burden
  penalty, with/without context features
- Cross-dataset generalisation: same headline comparison on All of
  Us, UK Biobank if data access is granted

These are explicitly *not* part of the pre-registered headline. They
are reported in a labelled "Exploratory" section of the paper.

---

## 7. Reporting standards

- All tables follow the framework's standard schema: `metric, value,
  CI_low, CI_high, n, p_value, effect_size`
- All figures include sample size, seed count, and the exact baseline
  set in the caption
- All metric names use plain English where the audience is
  non-expert (e.g. "Daily steps" not "step_aggregate_daily") and
  internal jargon where the audience is engineering
- All scripts, configs, and seeds are committed and DOI-archived
  before submission
- A CONSORT-style flow diagram is included for the simulated cohort
  (excluded / included / analysed) — adapted for simulation
  studies per latest Nature Methods guidance

---

## 8. Missing data and dropout

The simulator produces no missing data by construction. **However:**

- *Wearable dropout* (user stops wearing the device) is modelled in
  the user simulator as a state in the latent "stable" archetype. We
  report sensitivity to dropout rate (0%, 5%, 10%, 20%) as a
  pre-registered robustness check.
- *Simulation non-convergence* (RL agent fails to learn) is reported
  per (agent, seed). A seed is excluded from the headline if the
  agent's reward is below the random baseline by >1 SD at episode
  end. Exclusion rate is reported; results with and without
  exclusions are both shown.

---

## 9. Pre-registration and amendment log

| Version | Date | Change | Author |
|---|---|---|---|
| v0.1 | 2026-06-14 | Initial draft | W. Dennis (research-assistant prep) |

**Open questions to resolve before locking v1.0:**

- [ ] Mengyan: confirm clinical priority of cumulative steps vs.
      sustained change as primary outcome
- [ ] Mengyan: confirm acceptable subgroup gap threshold for the
      equity analysis (currently undefined)
- [ ] Swapnil: confirm whether the proxy-value iteration step in
      PR #85 counts as a pre-registered component or an ablation
- [ ] External: confirm the CONSORT-for-simulation reporting standard
      is acceptable to Nature Methods (no canonical standard exists
      yet)

---

## 10. Protocol deviation log

(Empty at v0.1. Filled when deviations occur.)

---

## 11. Pointers for the implementation team

These are *research-side* notes, not implementation tasks. They will
inform M-08 (Evaluation Framework) once the design doc is signed off.

- The headline metric (cumulative steps over 42 days) requires
  running each agent for a full 42-day episode per participant.
  Storage: per-participant daily step series + per-seed aggregate.
- Bootstrap replicates (B=1000) are computed within each run, not
  across runs. This is *separate* from the K=30 seed-based CI.
- The exclusion criterion for non-converging seeds (§8) is *not* in
  the current code design. It needs to be added to the
  ExperimentRunner spec.
- The framework's standard metric schema (`metric, value, CI_low,
  CI_high, n, p_value, effect_size`) is not yet formalised in
  `docs/code/`. This plan proposes it; the spec doc should adopt
  it explicitly.

---

## 12. What this plan does not decide

- **Safety constraint design.** That is a separate decision tree
  (forthcoming). Safety constraints interact with sample size —
  agents that violate constraints are excluded from the headline —
  but the constraint *thresholds* are a different conversation.
- **Online vs offline RL framing.** That is decision tree PR-C.
- **Reward shaping weights.** The framework exposes the
  burden-penalty weight λ as a config; the *value* of λ to use in
  the headline comparison is an open question that this plan defers
  to M-08.
- **Real-data integration.** Phase 2 will have its own analysis plan
  keyed off the data use agreement.

---

*End of Statistical Analysis Plan v0.1*
