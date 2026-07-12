---
title: "Framework Comparison — RL-driven Health Intervention Simulators"
status: "draft v0.1"
date: "2026-06-14"
author: "W. Dennis (research-assistant prep)"
purpose: "Positioning table for the Nature Methods paper introduction"
related: "docs/design/initial_design.tex §2 · docs/research/decision-trees/online-vs-offline-rl.md"
---

> **Archived 2026-06-23.** This document was part of the June-14 research batch
> (PRs #86–#90). It is retained for historical reference but is no longer actively
> maintained. See `docs/research/README.md` for current research artifacts.

# Framework Comparison — RL-driven Health Intervention Simulators

> A 2-axis positioning table for the paper's "why our framework"
> section. Compares rl-health-interventions against the closest
> published alternatives on two dimensions: (1) data source
> (synthetic / real / hybrid) and (2) configuration model
> (config-first / code-first). Plain-English labels throughout.

---

## 1. Frameworks surveyed

The audit flagged that the design doc mentions StepCountJITAI and
Health Gym in passing, with no positioning table. The table below
expands to 6 frameworks — the two cited plus four additional
analogues that an external reviewer is likely to flag as missing.

| # | Framework | Year | Primary use | Type |
|---|---|---|---|---|
| F1 | StepCountJITAI | 2024 | PA JITAI sim (single dataset) | Code-first, synthetic-data-first |
| F2 | Health Gym | 2023 | Synthetic clinical RL envs (MIMIC-III) | Code-first, real-data-derived |
| F3 | rl-health-interventions (ours) | 2026 | Configurable PA JITAI sim | Config-first, synthetic-then-real |
| F4 | MicroRCT simulator (Liu et al. 2024, internal reproduction) | 2024 | MRT power-analysis | Code-first, real-data-driven |
| F5 | CASIE (Computing, Simulation, Inference in mEdicine) | 2023 | Sepsis/HIV RL benchmarks | Code-first, real-data-derived |
| F6 | OHDSI patient-level prediction simulator | 2022 | EHR-based policy eval | Code-first, real-data-derived |

F4-F6 are *candidates for inclusion* — not currently cited in the
project. They are added because a Nature reviewer is likely to ask
"why not X?" and the answer should be in the paper, not invented in
rebuttal.

---

## 2. Two-axis positioning table

| Framework | Data source | Config model | Open source | Configurable user sim | Real-data calibration | Multiple datasets in one tool | PA-specific |
|---|---|---|---|---|---|---|---|
| F1: StepCountJITAI | Synthetic (single synthetic dist.) | Code (Python env class) | Yes (GitHub) | No (single behaviour) | No (synthetic only) | No (one dataset) | Yes (PA only) |
| F2: Health Gym | Real (MIMIC-III) | Code (Python env class) | Yes (GitHub) | No (no user sim) | Yes (MIMIC) | No (MIMIC only) | No (clinical, not PA) |
| F3: **rl-health-interventions (ours)** | **Synthetic first, real later** | **Config-first (YAML)** | **Yes (planned)** | **Yes (4 archetypes, configurable)** | **Phase 2 (HeartSteps)** | **Yes (config-driven)** | **Yes (PA, extensible)** |
| F4: MicroRCT sim | Real (single study) | Code (R/Python) | Varies | No | Yes | No | No |
| F5: CASIE | Real (MIMIC) | Code (Python) | Yes (GitHub) | No | Yes | No | No |
| F6: OHDSI PLP sim | Real (OMOP-CDM) | Code + SQL | Yes (GitHub) | No | Yes | Partial (any OMOP DB) | No |

**Axis 1 — Data source:** synthetic-only, real-only, or hybrid. The
project's hybrid (synthetic in Phase 1, real in Phase 2) is the
*only* row with that combination.

**Axis 2 — Configuration model:** config-first (YAML/JSON drives
behaviour, no code change needed) vs. code-first (Python class
subclassing). The project is the *only* row with config-first.

The cell of interest to Nature — config-first *and* hybrid data
*and* PA-specific *and* configurable user simulation — is empty in
the literature. That is the project's positioning claim.

---

## 3. Per-framework 100-word commentary

The 100-word limit is for the paper's introduction. Each entry is
drafted to be drop-in to a "Related work" paragraph.

### F1: StepCountJITAI (Karine 2024)

> StepCountJITAI is the closest published analogue to our work: a
> Python simulation environment for PA JITAIs built on synthetic
> step distributions. Its strength is reproducibility — one
> install, one experiment. Its limit is single-dataset scope and a
> fixed (non-configurable) user model. Researchers who want to
> compare two algorithms on the *same* simulator will find
> StepCountJITAI ready-to-use. Researchers who want to *change* the
> simulator — different user archetypes, different reward
> structures, different datasets — must fork the code. We position
> our framework as a config-first alternative that supports the
> same head-to-head comparisons without forking.

### F2: Health Gym (Gateno 2023)

> Health Gym provides synthetic RL environments derived from
> MIMIC-III for clinical decision support (sepsis, HIV,
> hypotension). It is open source and well-documented, and the
> trajectory format is directly transferable to PA settings. Its
> scope is clinical, not behavioural — there is no user simulator,
> no burden model, no engagement dynamics. We position our
> framework as the PA-specific, behaviour-focused counterpart:
> Health Gym for clinical RL, our framework for health-behaviour
> RL.

### F3: rl-health-interventions (ours)

> We present a config-first simulation framework for PA JITAIs.
> Unlike StepCountJITAI (single dataset) and Health Gym (clinical
> focus), our framework exposes the dataset schema, MDP dynamics,
> user behaviour model, and agent configuration as YAML — enabling
> cross-dataset and cross-policy comparisons without source-code
> changes. Phase 1 uses a synthetic user simulator with four
> behavioural archetypes; Phase 2 calibrates against HeartSteps V2
> data (and any other accessible wearable cohort). The framework
> is open source and reproducibility-first: every experiment is
> specifiable, runnable, and verifiable from a single config file
> plus a seed.

### F4: MicroRCT simulator (Liu et al. 2024, internal)

> Internal-use MicroRCT simulators exist for power-analysis
> purposes in clinical trial design. They are typically code-first,
> single-study, and not designed for RL agent evaluation. We do
> not position against them directly; they are mentioned for
> completeness in the related-work section.

### F5: CASIE

> CASIE is an open-source benchmark suite for offline RL
> algorithm development, with environments derived from MIMIC
> data. Its strength is standardisation — the same suite is used
> across many offline-RL papers. Its limit is clinical focus
> and no user-level behaviour model. We position as
> behaviour-focused: CASIE is the closest analogue for the
> *algorithm* dimension, StepCountJITAI for the *domain*
> dimension, and our framework combines both into a
> config-first PA-specific package.

### F6: OHDSI patient-level prediction

> OHDSI provides a large ecosystem of patient-level prediction
> tools on the OMOP common data model. These are observational,
> not RL. Mentioned only because reviewers from the clinical
> informatics community will ask "why not OMOP?" The answer is
> scope: OMOP is for observational prediction, not
> sequential-decision evaluation. Our framework could
> interoperate with OMOP cohorts as data sources in Phase 2 but
> is not a competitor.

---

## 4. What the table does NOT claim

- **That config-first is universally better.** Config-first is
  better for *researchers running comparative experiments*. It is
  not better for *framework authors* (writing config loaders is
  more work than writing a class) or for *production deployment*
  (where code-first is more flexible). The paper should be
  careful to claim the right audience.
- **That hybrid data is novel.** Many systems use synthetic data
  for development and real data for evaluation. What is novel is
  the *seamless config-driven transition* between the two.
- **That we beat any framework on any single axis.** The
  positioning is the *combination* of axes, not a per-axis
  comparison.

---

## 5. Open questions to resolve before locking v1.0

- [ ] Confirm F4-F6 are accurate. CASIE and OHDSI are
      best-guess entries; a literature check is needed.
- [ ] Decide whether to position against MicroRCT (F4). The
      project may not benefit from naming it.
- [ ] Decide on the inclusion order in the paper. The current
      order (StepCountJITAI, Health Gym, ours) follows the audit;
      an alternative is to organise by data type (synthetic first,
      real second, hybrid last).
- [ ] Confirm the framework's "config-first" claim is defensible.
      The audit notes that config-first is *designed but not
      implemented* in the codebase. The paper should say "config-
      first *architecture*" not "config-first *implementation*"
      until M-01 (Config Schema) is merged.

---

## 6. Sources

Frameworks referenced in this document, with their primary citation
and source code (where applicable):

| Framework | Primary citation | Source code |
|---|---|---|
| StepCountJITAI | karine2024stepcountjitai (in `docs/design/references.bib`) | github.com/karinekay/StepCountJITAI (verify) |
| Health Gym | gateno2023healthgym (in `docs/design/references.bib`) | github.com/nikooo777/health-gym |
| MicroRCT sim | Liu et al. 2024 (not in bib — needs entry) | varies |
| CASIE | Tang et al. 2023 (not in bib — needs entry) | github.com/clinicalml/casie (verify) |
| OHDSI PLP | OHDSI community (not in bib — needs entry) | github.com/OHDSI |

---

*End of Framework Comparison v0.1*
