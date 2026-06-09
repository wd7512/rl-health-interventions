# rl-health-interventions: Codebase Plan

**Date:** 2026-06-09
**Context:** Bristol x NUS summer internship, 8-week foundation phase
**Stakeholders:** William Dennis (builder), Mengyan Zhang (Bristol data), Swapnil Mishra (NUS/MLGH)
**Repo:** `rl-heath-interventions`

---

## What This Is

A configurable simulation framework for testing RL-driven health interventions on wearable device data. Swapnil's team plugs in their datasets, specifies the MDP and hypotheses via config files, and runs experiments comparing intervention policies.

**Phase 1:** Foundational framework — config-driven data layer, MDP environment, rule-based user simulation, RL agent library, experiment runner.

**Phase 2 (stretch):** LLM-based user simulation, validation against real data, LLM-augmented experiments.

The framework is the platform. The 5 gaps from the literature review are experiments run *on* the platform, not part of it.

---

## Done Criteria (8-Week Internship)

Software-oriented deliverables:

1. A working CLI: `uv run rl-health --config experiment.yml` that trains agents and produces a results table
2. Config schema documented with an example for the proposed MDP (google doc)
3. A README that lets Swapnil clone, `uv sync`, and run the same experiment
4. Config schema is stable across 2+ different dataset schemas and 2+ MDP designs (stretch)

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
- **Config format:** TBD (JSON or YAML — to confirm with supervisors)
- **Dependencies:** Minimal. No SB3 or Gymnasium unless proven necessary
- **Interface:** CLI + config files
- **Package:** TBD (installable or repo-based — to confirm with supervisors)

---

## Open Decisions (Pending Supervisor Input)

1. JSON or YAML for config?
2. CLI vs notebook vs web — any strong preferences?
3. Installable package or repo-based?
4. Does the proposed MDP look like a reasonable starting point?

---

## Literature Review Reference

Full review at `docs/literature-review/`.

5 major gaps identified (experiments *on* the framework, not *in* it):
1. No LLM simulation validated for wearable health data
2. No delayed-feedback bandit for 3-week clinical measures
3. No non-stationary adaptation for PA interventions
4. No standardised MDP/benchmark for PA JITAI RL
5. No offline RL for PA interventions with real wearable data
