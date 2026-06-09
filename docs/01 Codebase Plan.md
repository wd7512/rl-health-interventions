# rl-health-interventions: Codebase Plan

**Date:** 2026-06-09
**Context:** Bristol x NUS summer internship, 8-week foundation phase
**Stakeholders:** William Dennis (builder), Mengyan Zhang (Bristol data), Swapnil Mishra (NUS/MLGH)
**Repo:** `rl-health-interventions`
**MDP Design Doc:** `docs/02 MDP Specification.tex` (formalised from google doc)

---

## What This Is

A configurable simulation framework for testing RL-driven health interventions on wearable device data. Swapnil's team plugs in their datasets, specifies the MDP and hypotheses via config files, and runs experiments comparing intervention policies.

**Phase 1:** Foundational framework — config-driven data layer, MDP environment, rule-based user simulation, RL agent library, experiment runner.

**Phase 2 (stretch):** LLM-based user simulation, validation against real data, LLM-augmented experiments.

The framework is the platform. The 5 gaps from the literature review are experiments run *on* the platform, not part of it.

---

## Done Criteria (8-Week Internship)

Software-oriented deliverables:

1. A working CLI: `uv run rl-health-interventions --config experiment.yml` that trains agents and produces a results table
2. Config schema documented with an example for the formal MDP spec (Overleaf design doc)
3. A README that lets Swapnil clone, `uv sync`, and run the same experiment
4. Config schema is stable across 2+ different dataset schemas and 2+ MDP designs (stretch)
5. Public dataset feasibility report: evaluate All of Us and UK Biobank for simulator training

Papers run *on* the framework later. Phase 2 (LLM simulation) is stretch — core deliverable is working infrastructure.

---

## Architecture

### Config-First Design

The config schema is the load-bearing wall. Every component reads from config:
- Data schema mapping (column names → semantic fields)
- MDP specification (states, actions, rewards, transitions)
- Agent hyperparameters
- Experiment definitions (what to compare, what to measure)

New dataset = new config, not new code. Genuinely new data structures (e.g., GPS, free text) may require a code bump with guidelines.

### Phased Structure

**PHASE 1: FOUNDATIONAL FRAMEWORK**

*Subphase 1A: Config-Driven Data Layer*
- Configurable data schema (map columns → semantic fields)
- Ingest pipeline that works for any wearable dataset given a config
- Feature engineering pipeline (also config-driven)
- Synthetic data generator with configurable properties

*Subphase 1B: Config-Driven MDP Environment*
- Entire MDP spec from config: states, actions, transitions, rewards
- Multi-timescale reward support (immediate steps + delayed body measures)
- Pluggable transition models (rule-based initially)
- Minimal environment interface

*Subphase 1C: User Simulation Engine (Rule-Based)*
- Configurable user profiles and response models
- Behavioural theory grounding (goal-driven, social responder, resistant, etc.)
- Backlash/fatigue mechanisms
- Validation framework

*Subphase 1D: Config-Driven Agent Library*
- Agents parameterised from config
- Baselines: Thompson Sampling, ε-greedy, LinUCB
- RL: DQN, Double DQN, PPO
- Offline: CQL, IQL
- All share common interface
- Minimise dependencies — implement from scratch where reasonable

*Subphase 1E: Experiment Runner & Results*
- Config → train → compare → report
- Statistical comparison, visualisation, export
- Reproducibility (seeds, config snapshots)

**PHASE 2: LLM SIMULATION (Stretch)**

- LLM-as-user architecture (prompt-driven persona simulation)
- Calibration against real wearable data distributions
- Statistical comparison: LLM-simulated vs real user behaviour

---

## Proposed MDP (from google doc)

**State:** steps, heart rate, sleep, time of day, past response history, goal progress, user profile

**Actions (discrete):** no message, motivational prompt, walking suggestion, goal reminder, recovery suggestion, progress feedback

**Reward:** `R_t = α·Δsteps - β·notification_burden + λ·goal_progress`, plus delayed body measures every 3 weeks

**Discount factor:** γ ≈ 0.9–0.95

---

## Tech Stack

- **Language:** Python 3.11
- **Package manager:** `uv`
- **Config format:** YAML (confirmed with Mengyan)
- **Dependencies:** Minimal. No SB3 or Gymnasium unless proven necessary
- **Interface:** CLI + config files (web UI = stretch goal, not necessary at this stage)
- **Package:** TBD (installable or repo-based — to confirm with supervisors)

---

## Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Config format: YAML | Confirmed (Mengyan) |
| 2 | Interface: CLI + config files; web UI = stretch | Confirmed (Mengyan) |
| 3 | Installable package vs repo-based | Pending |

## Open Decisions (Pending Supervisor Input)

1. Installable package or repo-based?
2. Does the proposed MDP look like a reasonable starting point? (awaiting Swapnil)

---

## Public Datasets (Feasibility Complete ✅)

Two datasets evaluated for simulator training. Full study: [`docs/03 Data Sources.md`](03%20Data%20Sources.md).

1. **All of Us Fitbit Dataset** — Nature Medicine 2026
   - 59,000+ participants, Fitbit data, 14-year span
   - 39M+ step observations, 31M+ sleep observations
   - 46% linked to EHR, physical measurements, genomics
   - Access: All of Us Research Program researcher workbench (cloud-only, no local download)
   - [Paper](https://www.nature.com/articles/s41591-026-04352-3)

2. **UK Biobank Accelerometer Dataset** — npj Digital Medicine 2024
   - 700,000+ person-days, wrist-worn tri-axial accelerometer
   - 100,000+ participants, 7 days free-living data each
   - Pre-trained SSL models available (OxWearables)
   - Access: UK Biobank application (project ref required, 4-8 weeks)
   - [Paper](https://www.nature.com/articles/s41746-024-01062-3) | [Code](https://github.com/OxWearables/ssl-wearables)

**Finding:** Both datasets require institutional applications with 4-8 week lead times. Neither provides downloadable samples. **Phase 1 uses synthetic data generators parameterised from published statistics.** Real data integration is Phase 2.

**Supplementary finding:** HeartSteps V1/V2 micro-randomized trials contain the only available *intervention response* data. These are smaller (50–100 participants) but directly calibrate the user simulation (1C). See [`docs/04 Additional Data Sources.md`](04%20Additional%20Data%20Sources.md).

## Literature Review Reference

Full review at `C:\Obsidian_Vaults\main\10 Research\Bristol x NUS RL\Literature Review\05 Master Literature Review.md`.

5 major gaps identified (experiments *on* the framework, not *in* it):
1. No LLM simulation validated for wearable health data
2. No delayed-feedback bandit for 3-week clinical measures
3. No non-stationary adaptation for PA interventions
4. No standardised MDP/benchmark for PA JITAI RL
5. No offline RL for PA interventions with real wearable data
