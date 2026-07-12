---
title: "Research front summary — for supervisor review"
status: "draft v0.1"
date: "2026-06-14"
audience: "Primary supervisor + Mengyan (Bristol) + Swapnil (NUS)"
purpose: "What 5 research artefacts now exist, what they let us claim, what still blocks the paper"
---

> **Archived 2026-06-23.** This document was part of the June-14 research batch
> (PRs #86–#90). It is retained for historical reference but is no longer actively
> maintained. See `docs/research/README.md` for current research artifacts.

# Research front summary

> A supervisor-facing summary of the 5 research artefacts opened as
> PRs on 2026-06-14. Each PR is a self-contained chunk of value;
> none modify code, the design doc, or main.

## The 5 PRs

| # | PR | Branch | What it lets us claim |
|---|---|---|---|
| 86 | Statistical analysis plan | `research/stat-analysis-plan` | We have a pre-registered evaluation methodology (primary outcome, baselines, sample size, inference procedure) |
| 87 | Simulator validation strategy | `research/simulator-validation` | We know how we'll validate the user simulator in Phase 1 and Phase 2 |
| 88 | Online vs offline RL framing | `research/online-vs-offline-rl` | We have a clear, defensible paper framing (simulation-based policy evaluation) |
| 89 | Framework comparison | `research/framework-comparison` | We can write a "why our framework" section that positions us in an empty cell of the literature |
| 90 | PR #85 review | `research/pr-85-review` | We have a documented code-and-results review of our one real-data artefact |

## How the 5 artefacts fit together

```
            ┌─── PR #86 (statistical analysis plan) ───┐
            │   "How we evaluate"                       │
            │                                           │
PR #88 ─────┤   PR #87 (simulator validation) ────────┤── PR #89 (framework comparison)
"Paper      │   "How we know the simulator is OK"      │   "How we position vs. the literature"
 framing"   │                                           │
            │   PR #90 (PR #85 review) ────────────────┤
            │   "Reality check on the one real artefact"│
            └───────────────────────────────────────────┘
```

**Reading order for a new reviewer:** #88 first (the framing), then
#86 (the metrics), then #87 (the validation), then #89 (the
positioning), then #90 (the PR #85 sanity check).

## What this lets us claim, by area

### Methodology
- We have a pre-registered primary outcome (cumulative steps,
  H1: ≥10% gain over best baseline, d≥0.3)
- We have a frozen baseline set (random, fixed, rule-based,
  contextual bandit, TS, HeartSteps V2)
- We have a frozen sample size (N=200, K=30, B=1000)
- We have a frozen inferential procedure (paired t with BH-FDR
  q=0.05, Cohen's d)

### Positioning
- We can claim the empty cell in the literature:
  config-first + hybrid data + PA-specific + configurable user
  simulation
- We can draft a 100-word positioning paragraph per competing
  framework (StepCountJITAI, Health Gym, MicroRCT sim, CASIE,
  OHDSI)

### Framing
- We can claim "simulation-based policy evaluation" as the
  paper's primary contribution
- We can say with confidence that Phase 1 alone is publishable
  as a framework paper, de-risking the timeline
- We have a clear role for PR #85 in the story: it is the Phase 2
  offline-RL prototype, not a numerical reproduction of the
  paper

### Validation
- We have a two-stage validation strategy
- We have a pre-registered noise floor (10% on NHANES
  fingerprints, 2× replay MSE on HeartSteps V2)
- We have a fallback if HeartSteps V2 access is delayed (4TU #1
  "Collaboratively Setting Daily Step Goals", with caveats)

### Reproducibility
- We have a documented review of the one existing real-data
  artefact (PR #85). The review identifies two blockers
  (provenance discrepancy, episode-length gap) and a recommended
  fix.

## What still blocks the paper

In strict priority order:

1. **PR #85 fix (provenance + episode length).** Blocked on the
   author (you) running the canonical config and updating
   REPORT.md. PR #90 has the drop-in review comment.

2. **Swapnil's MDP sign-off (#48, decision 3).** This is the
   existing external blocker. The framing in PR #88 and the
   validation in PR #87 are designed to be *MDP-agnostic* — they
   don't require a specific MDP sign-off to be useful, but the
   exact baseline set in PR #86 may need adjustment once the MDP
   is locked.

3. **Mengyan's clinical-priority confirmation (PR #86, open
   question 1).** Cumulative steps vs. sustained change as
   primary outcome. Affects the headline metric but not the
   rest of the analysis plan.

4. **Mengyan's equity-threshold confirmation (PR #86, open
   question 2).** Affects the exploratory subgroup analysis only.

5. **Phase 2 data access (PR #87, open question 1).** HeartSteps
   V2 data request is in flight; 4TU #1 is the documented
   fallback. Affects the Stage 2 validation gate.

6. **External review of the analysis plan (PR #86, open question
   3).** A 1-hour review by an external RL-for-health researcher
   would close the "no external review process exists" gap from
   the audit.

## What this does NOT claim

- The research artefacts do not implement anything. Code is
  blocked on the design doc sign-off.
- The research artefacts do not modify the design doc. The
  framing recommendation in PR #88 is a *future supplementary
  document*, not an edit to the gated design doc.
- The research artefacts do not modify the roadmap. They are
  inputs to the design conversations that the roadmap tracks.
- The research artefacts do not pre-empt Swapnil's MDP sign-off
  or Mengyan's clinical-priority confirmation. They are
  drafts awaiting those decisions.

## What to do next

This is a 1-week action list. None of it is implementation.

1. **You:** respond to the PR #85 review comment in PR #85's
   thread. The drop-in comment is in PR #90 §7.
2. **You:** decide which of the 5 PRs to send to Mengyan and
   Swapnil first. Suggested: PR #88 (framing) to Swapnil, PR #86
   (analysis plan) to Mengyan.
3. **You (or me, on request):** draft the paper-framing
   supplementary document recommended in PR #88. This is the
   first concrete deliverable that uses the research front.
4. **You (or me, on request):** identify one external RL-for-
   health researcher to review PR #86. The audit flagged that
   no external review process exists; this is the seed.
5. **Future chunks** (deferred — see the master plan): safety
   constraint design, clinical-validity metrics, IRB pre-
   registration, reward shaping, HeartSteps V2 citation
   verification.

## What this summary is not

- Not a paper draft. The research artefacts are *inputs* to a
  paper, not the paper itself.
- Not an implementation plan. See `docs/plans/ROADMAP.md` and the
  subphase docs in `docs/plans/stale/` for that.
- Not a replacement for the design doc. The design doc is the
  source of truth on the MDP, the user simulator, and the
  agent library. These research artefacts depend on those
  decisions being made; they don't make them.

---

*End of Research front summary v0.1*

**Master plan:** `.hermes/plans/2026-06-14_research-assistant-plan.md`
**PR list:** #86, #87, #88, #89, #90 (all open, no main-branch changes)
