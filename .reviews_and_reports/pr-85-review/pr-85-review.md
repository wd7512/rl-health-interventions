---
title: "PR #85 Review — HeartSteps V2 Paper Reproduction"
status: "research review v0.1"
date: "2026-06-14"
reviewer: "W. Dennis (research-assistant prep, no conflict of interest — author and reviewer are the same person, this is a pre-merge sanity check)"
target: "PR #85 — feat: HeartSteps V2 paper reproduction with real NHANES data"
scope: "Algorithm faithfulness, results validity, methodology assumptions, suitability as a paper-reproduction artefact"
related: "PR #85 body · paper_reproduction/PLAN.md · paper_reproduction/results/REPORT.md · docs/research/decision-trees/online-vs-offline-rl.md"
---

# PR #85 Review — HeartSteps V2 Paper Reproduction

> Pre-merge review of the HeartSteps V2 reproduction PR. The
> findings here will (a) inform the PR review comments and (b) feed
> the paper's methodology section. The author and reviewer are the
> same person; the review is a sanity check, not an external
> assessment.

---

## 1. Headline finding (must fix before merge)

**The PR body and REPORT.md report different headline numbers.**

| Source | % improved | Mean improvement | Participants | Days | Re-runs |
|---|---|---|---|---|---|
| PR body | 60.0% (18/30) | +2.32 | 30 | 7 | 5 |
| REPORT.md | 46.7% (7/15) | -1.91 | 15 | 30 | 10 |
| `run_simulation` defaults | — | — | 30 | 90 | 10 |

Neither the PR body nor REPORT.md matches the function defaults.
Neither number can be reproduced from the code without running the
specific (n_participants, n_days, n_re_runs) tuple, which is not
recorded in either document.

**This is a blocker for the PR being mergeable into a paper-reproduction
context.** The paper's methodology section cannot cite a result whose
provenance is unrecorded.

**Suggested fix:**

1. Add a `results/runs/` directory with one JSON per (config, seed)
   run, including the exact CLI invocation that produced it.
2. Pin a single canonical run configuration and report only those
   numbers in the PR body, REPORT.md, and the paper.
3. Add a CI step that re-runs the canonical configuration and
   checks the result file is updated.

---

## 2. Algorithm faithfulness to Liao et al. 2019

The reproduction claims to implement Sections 5.2–5.4 of the paper.
Reviewed each module against the paper's equations.

### 2.1 Dosage variable (Section 5.2) — `dosage.py`

**Paper equation 1:** `X_{t+1} = lambda * X_t + 1{E_t}` where
`E_t` is the event indicator.

**Code (`dosage.py:74`):** `self._value = self.decay * self._value + event_int`

**Verdict:** Faithful. Decay default of 0.95 matches paper. The
treatment and anti-sedentary events are combined into a single
event_int — that matches the paper's E_t = 1 if *either* event
fires (paper section 5.2 paragraph 2).

**Minor concern:** The paper describes the event indicator as
depending on treatment *and* anti-sedentary messages. The code
implements this. The PR body's "anti-sedentary dosage bug" claim
suggests an earlier version didn't — that bug-fix history is
preserved in the commit log but not in the docstring. Consider
adding a "Fixed in commit X" note.

### 2.2 Bayesian linear regression (Section 5.4.1) — `bayesian_regression.py`

**Paper equations 5-6:**

```
phi(S, A) = [g^T, pi * f^T, (A - pi) * f^T]
Sigma_{d+1} = (sum phi*phi^T / sigma^2 + Sigma_prior_inv)^{-1}
mu_{d+1} = Sigma_{d+1} * (sum phi*R / sigma^2 + Sigma_prior_inv * mu_prior)
```

**Code (`bayesian_regression.py:162-176`):** Implements this
exactly. Construct_features at line 132-134 is correct.
Posterior update at 166-173 is correct.

**Verdict:** Faithful.

**Concern (medium):** Line 167 — `np.linalg.inv(self._posterior_cov)`.
The PR body mentions switching to `pinv` for OLS, but this line still
uses `inv`. If `Sigma_prior` is ever near-singular, this is a
numerical risk. The Bayesian posterior *should* be well-conditioned
by construction (added outer products on top of prior inverse), so
this is low-risk in practice — but worth flagging.

### 2.3 Thompson Sampling (Section 5.3) — `agent.py`

**Paper:** Sample beta from posterior, compute Q-values, softmax,
clip to [epsilon_1, epsilon_0], sample.

**Code (`agent.py:93-123`):** Sample beta (line 93), compute Q0/Q1
(lines 102-114), softmax (lines 116-118), clip (line 120), sample
(line 121).

**Verdict:** Faithful. Clipping to [0.1, 0.2] matches paper
Section 5.3.

**Concern (low):** The Q-value formula (lines 102-114) computes
Q0 and Q1 including the proxy-value term. In the paper, the
delayed effect is added as `gamma * H(x, a)`. The code does this
(lines 107, 113). However, the action-centring term
`(A - pi) * f^T beta` is split between the two Q-values: Q0
gets `(0.0 - pi) * f^T beta` and Q1 gets `(1.0 - pi) * f^T beta`.
This is correct (the action-centring form is a notational
shorthand) but worth a code comment.

### 2.4 Proxy value (Section 5.4.2) — `proxy_value.py`

**Paper:** Simplified MDP over (dosage x, availability i). Value
iteration on discretised grid. H(x, a) = E[V*(x', i') | x, a].
Blending weight w in [0, 1] between H_1 and H*.

**Code (`proxy_value.py`):** Implements this. The Bellman update
(lines 144-175) and the value-iteration loop (lines 188-203) look
correct.

**Concern (medium):** The reward function `_reward` (lines 92-114)
models `r(x, 1) = treat_benefit * max(0, 1 - burden_coef * x)`.
This is a *modelling choice* — the paper's simplified MDP uses a
binary reward in the published version (Section 5.4.2). The
reproduction uses a continuous, dosage-dependent reward. This is
a *modification* of the paper, not an implementation. The
modification is documented in the PR body ("was penalising ALL
intervention regardless of dosage; fixed to model
dosage-dependent treatment benefit"), but the consequences for
the headline results are not analysed. A reviewer of the
*paper*, not the PR, will ask: is this a faithful reproduction
or an extension? The answer should be explicit in REPORT.md.

### 2.5 Nightly update — `nightly_update.py` (not reviewed in depth)

The file is 96 lines and follows the standard pattern (call
`proxy_value.update()`, blend posterior, log). No
show-stopping concerns from the size and shape; would need a
full read for confidence.

---

## 3. Results reproducibility

This is the most concerning section. Per §1, the numbers in the
PR body and REPORT.md do not match. Additionally:

- The `verification_report.md` mentioned in the file list was not
  reviewed here; it should be cross-checked against REPORT.md.
- The reproduction is reported as "60% improved" but the
  `simulate` defaults are 30/90/10 (n_participants/n_days/n_re_runs),
  which the PR body claims to use with 30/7/5 — different days and
  re-runs. The 7-day episode is *not* long enough to exercise the
  dosage decay (decay=0.95 means dosage from day 1 has weight
  0.95^6 ≈ 0.74 at day 7, but most of the design-of-experiment
  signal in the paper comes from day 14+).
- The paper itself uses 42 days. Using 7 days fundamentally
  changes what the experiment measures — the dosage variable
  doesn't have time to plateau, and the proxy-value iteration
  is not really exercised.

**Recommendation:** Re-run the canonical experiment at the paper's
42 days, report those numbers in the PR body and REPORT.md, and
note any deviation from 42 days as a sensitivity analysis.

---

## 4. Methodology assumptions worth flagging for the paper

### 4.1 Prior circularity

REPORT.md notes: "Our prior is constructed from the same
generative model used for evaluation." The paper's prior is
constructed from *independent* HeartSteps V1 pilot data. This is a
fundamental difference. The reproduction is therefore a *sanity
check* that the algorithm works on synthetic data, not a
replication of the paper's headline finding.

For the project, this is fine as a Phase 1 deliverable. For the
paper, it needs to be framed honestly: "We validate the algorithm
on synthetic data; a Phase 2 replication on real HeartSteps V1
data is the natural next step."

### 4.2 Sample size

Paper: 37 participants, 96 re-runs. Reproduction REPORT: 15
participants, 10 re-runs. PR body: 30 participants, 5 re-runs.
Neither matches the paper. The analysis plan we just wrote
(`statistical-analysis-plan.md`) calls for 200 participants, 30
seeds, 1000 bootstraps. There's a large gap between the
reproduction's compute and the analysis plan's requirement.

**Recommendation:** Phase 2 should re-run this experiment at
analysis-plan-grade scale (200 participants minimum).

### 4.3 Features

Paper: 10+ features (location, temperature, app engagement, time
of day, etc.). Reproduction REPORT: 4 + 2. PR body (after commit
e882e0c "enriched feature set"): 5 + 4. The enrichment is a
modification, not a reproduction. State explicitly in the paper
what was added and what was dropped.

---

## 5. Test coverage

PR body claims "104+ unit tests covering all modules. Full test
suite passes. Ruff clean." I did not run the test suite in this
review (out of scope for a research review). The test file list
shows 12 test files covering all 5 algorithm modules + 4
simulation modules + 1 baseline + 1 visualization. The test
counts in each file (per the PR diff) suggest the 104+ claim is
plausible.

**Recommendation:** A second reviewer (not the author) should run
`uv run pytest tests/ -q` and `uv run ruff check` to confirm.

---

## 6. What the reproduction is good for (and what it isn't)

**Good for:**

- Demonstrating that the HeartSteps V2 algorithm can be
  implemented from the paper equations alone (no source code
  from the original authors).
- End-to-end pipeline testing: data → features → agent → env →
  results.
- A code artefact that the paper can point to ("our reproduction
  is at github.com/.../paper_reproduction").

**Not good for:**

- Claiming numerical reproduction of the paper's headline
  finding. The data is synthetic, the features are reduced, the
  prior is circular, and the run config doesn't match the paper.
- A standalone "reproduction" paper. The differences from the
  paper are too large for the standard "reproducibility study"
  framing. A better framing is "an open implementation
  validated on synthetic data, awaiting real HeartSteps V1 data
  for replication."

---

## 7. PR-review comment suggestions

Drop-in comment for the PR thread (write as a single review
comment, not 7 small ones):

> Thanks for the careful reproduction work. The algorithm
> implementation looks faithful to the paper (Sections 5.2–5.4).
> Two blockers before this can support a paper-reproduction
> claim:
>
> 1. **Provenance.** The PR body and REPORT.md report different
>    headline numbers (60%/+2.32 vs 46.7%/-1.91), and neither
>    matches the `run_simulation` defaults. Please pin a single
>    canonical configuration, run it, save the invocation and
>    output under `results/runs/`, and update both the PR body
>    and REPORT.md to reference that file.
>
> 2. **Episode length.** The PR body uses 7-day episodes. The
>    paper uses 42 days. At 7 days, the dosage variable doesn't
>    reach steady state and the proxy-value iteration isn't
>    exercised. Please re-run at 42 days and report those
>    numbers.
>
> After those two changes, the reproduction can be cited as
> "validated on synthetic data; replication on real HeartSteps
> V1 is Phase 2 work." The current text overclaims.
>
> Minor:
> - `bayesian_regression.py:167` still uses `np.linalg.inv`. PR
>   body mentions switching to `pinv` — the `inv` call here is
>   probably fine by construction but worth a comment.
> - `proxy_value.py:_reward` uses a continuous
>   dosage-dependent reward, not the paper's binary reward.
>   Please document this as a deliberate modification in
>   REPORT.md.
> - Consider adding a "What this reproduction is good for and
>   what it isn't" section to REPORT.md. I drafted one in
>   `docs/research/pr-85-review.md`.

---

## 8. How this review connects to the rest of the research plan

- The **online-vs-offline-rl decision tree** (PR-C) recommends
  reframing the project as "simulation-based policy evaluation."
  The PR #85 reproduction fits naturally as the Phase 2 prototype
  under that framing — it is literally an offline-policy
  evaluation on synthetic logged data.
- The **statistical analysis plan** (PR-A) calls for 200
  participants, 30 seeds, 1000 bootstraps. The reproduction's
  15/10 numbers are a Phase 1 sanity check, not Phase 2.
- The **simulator validation decision tree** (PR-B) recommends
  a two-stage validation. The reproduction provides Stage 1
  validation evidence (algorithm works on synthetic data) but
  not Stage 2 (replay on real HeartSteps V1).

The three research chunks (A, B, C) and this review (E) form a
coherent research front: A says how to evaluate, B says how to
validate the simulator, C says what to claim, E is the
ground-truth check on the one piece of real data the project has
touched (NHANES).

---

## 9. Recommendation

**Do not merge PR #85 as-is.** Address the provenance
discrepancy (§1) and the episode-length gap (§3) first. After
those are fixed, the reproduction is a valuable Phase 1 artefact
and a useful anchor for the Phase 2 HeartSteps V1 replication
work.

The PR can be re-reviewed after the fixes. This review document
will be updated to v0.2 with the re-review.

---

*End of PR #85 review v0.1*
