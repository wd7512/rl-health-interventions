# rl-health-interventions — Research Assistant Plan

**Date:** 2026-06-14
**Mode:** Pre-implementation research (no code, no main merges, design-doc gate unchanged)
**Repo state:** 3.3/10 health, design doc awaiting Swapnil sign-off, paper repro PR #85 open
**Goal of this phase:** Get the project to a position where unblocking the design doc also unblocks a Nature-grade submission. Research chunks, not implementation.

---

## Constraints (do not violate)

1. No commits to `main`. All work on branches, all changes via PR.
2. No code generation. Research artefacts are markdown only.
3. No changes to `docs/design/initial_design.tex` (awaiting Swapnil's sign-off on the MDP).
4. No changes to `docs/plans/ROADMAP.md` (already merged, signed off in effect via #84).
5. No new stub implementations in `src/rl_health_interventions/`. The audit found 8/17 stubs; don't add more.
6. PRs must be self-contained and mergeable on their own. No "part 1 of 3" linking.
7. PRs touching `docs/research/` need a clear external reviewer in mind (supervisor, Mengyan, or external).

## Working conventions

- Research artefacts go in `docs/research/` (committed to repo so they're reviewable on GitHub).
- Decision-tree reports follow the schema in `docs/research/decision-trees/_template.md` (to be created in chunk 1).
- Each chunk = one feature branch + one PR + one research note.
- Branch naming: `research/<chunk-slug>` (e.g. `research/stat-analysis-plan`).
- Reports use plain language labels, not jargon (per user preference: "Speeches with Party" not "Actor Coverage").
- Save research scripts/notes from past work to skill if patterns emerge.

---

## Chunked research PRs (priority order)

### PR-A: Statistical analysis plan (HIGH)
**Why first:** The audit lists "Statistical analysis plan pre-specified" as a missing Nature requirement (#3 on the readiness checklist). This is also a *pre-registration* artefact — a research-paper-grade doc that supervisors expect to see *before* Phase 2 real-data work begins. Cheap to write, high external value, sets the metric conventions every later chunk will reference.
**Artefact:** `docs/research/statistical-analysis-plan.md`
**Contents:** Primary outcome (cumulative steps), secondary outcomes, baselines (random, fixed, rule-based, contextual bandit, TS), metrics (regret, adherence, equity gap), multiple-comparisons correction, sample size / power, bootstrap CIs, subgroup analyses, missing-data handling, pre-registered hypotheses vs exploratory.
**Estimated effort:** 6-8 hours of literature + writing. No code.
**External input needed:** Mengyan to confirm clinical outcome priorities.

### PR-B: Decision tree — simulator validation strategy (HIGH)
**Why second:** Phase 2 hinges on whether the simulator is good enough to draw conclusions from. Multiple valid validation strategies exist; the choice shapes data collection, metric design, and what claims the paper can defensibly make. The audit's "Reproducibility score: 6/10" comes mostly from this gap.
**Artefact:** `docs/research/decision-trees/simulator-validation.md`
**Decision axes:**
  - Behavioural fingerprint matching vs. real-data replay vs. regret-bound calibration vs. predictive checks
  - Internal validity only vs. external (cross-dataset) validity
  - Black-box simulator vs. interpretable parametrised simulator
**Output:** Recommended strategy with rationale, fallback, and what to publish vs. cite as future work.

### PR-C: Decision tree — online vs. offline RL framing (MEDIUM-HIGH)
**Why third:** The audit flagged "Offline RL for Phase 2 not motivated despite limited MRT data." The PR #85 work is *literally* offline RL (HeartSteps V2 paper). Without an explicit framing decision, the paper has no story for why it picks online or offline.
**Artefact:** `docs/research/decision-trees/online-vs-offline-rl.md`
**Decision axes:** Pure online (deployed JITAI) vs. pure offline (logged-data) vs. hybrid (offline pretrain + online fine-tune) vs. sim-to-real.
**Output:** Recommended framing with reference to StepCountJITAI, HeartSteps V2, and the project's NUS cohort study.

### PR-D: Comparison-to-existing-frameworks table (MEDIUM)
**Why fourth:** Nature needs positioning. The design doc mentions StepCountJITAI and Health Gym in one paragraph. A 2-axis table (synthetic/real × config/coded) covering ≥6 frameworks (StepCountJITAI, Health Gym, MicroRCT, CASIE, etc.) gives the paper a defensible "why our framework" section.
**Artefact:** `docs/research/framework-comparison.md`
**Output:** Table with axes, brief commentary, and a positioning paragraph draftable into the paper's introduction.

### PR-E: PR #85 review (MEDIUM)
**Why fifth:** PR #85 has zero reviewers. Reading 6,444 lines is too much for one session — but a targeted review of (a) the HeartSteps V2 algorithm faithfulness to the paper sections 5.2-5.4, (b) the bugs-found-and-fixed section, and (c) the test coverage, is a real research contribution. Findings go in `docs/research/pr-85-review.md` then become PR comments.
**Why research, not just review:** The review findings will inform the paper's methodology section (what the reproduction validates, what it doesn't, what the magnitude gap implies).
**External input needed:** None. Pure literature + code reading.

### Future chunks (NOT in this plan, will re-scope later)
- Decision tree: safety constraint design (burden thresholds, intervention limits, opt-out)
- Decision tree: clinical-validity metrics vs ML metrics
- Ethical/IRB pre-registration template
- Decision tree: reward shaping (immediate vs. delayed weighting)
- HeartSteps V2 citation verification (audit-blocked: waiting on publication check)

---

## What this plan deliberately does NOT do

- **Does not implement code.** All implementation is blocked on design-doc sign-off (#48, decision 3).
- **Does not propose changes to the MDP spec.** That's Swapnil's call.
- **Does not write the actual paper.** Research chunks produce the *inputs* a paper needs (analysis plan, positioning, validation strategy) — not the paper itself.
- **Does not duplicate the audit.** The 6-phase audit (PR #78) already produced a 30-item improvement list with priorities. This plan complements it; it doesn't redo it.
- **Does not babysit PR #85.** PR #85 is its own workstream. The research-assistant touches it only in chunk E.

---

## Done when

- 5 PRs (A-E) merged or open-with-review-comments
- A supervisor-facing summary doc that says: "Here are the 5 research artefacts. Here is what they let us claim. Here is what still blocks the paper."
- A short list of which research artefacts blocked the design-doc sign-off decision (i.e. did the statistical analysis plan help Swapnil evaluate the MDP?)
