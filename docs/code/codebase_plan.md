# rl-health-interventions: Codebase Plan

**Date:** 2026-06-09
**Context:** Bristol x NUS summer internship, 8-week foundation phase
**Stakeholders:** William Dennis (builder), Mengyan Zhang (Bristol data), Swapnil Mishra (NUS/MLGH)
**Repo:** `rl-health-interventions`
**MDP Design Doc:** `initial_design.tex` (formalised from google doc)

---

## What This Is

A configurable simulation framework for testing RL-driven health interventions on wearable device data. Swapnil's team plugs in their datasets, specifies the MDP and hypotheses via config files, and runs experiments comparing intervention policies.

**Phase 1:** Foundational framework — config-driven data layer, MDP environment, rule-based user simulation, RL agent library, experiment runner.

**Phase 2:** Real-data validation — calibrate the user simulator and ground MDP dynamics against observed behavioural responses using HeartSteps V1/V2, All of Us Fitbit Dataset, and UK Biobank Accelerometer Dataset. Beyond Phase 2, stretch goals include LLM-based user simulation.

The framework is the platform. The 5 gaps from the literature review are experiments run *on* the platform, not part of it.

---

## Done Criteria (8-Week Internship)

Software-oriented deliverables:

1. A working CLI: `uv run rl-health-interventions --config experiment.yml` that trains agents and produces a results table
2. Config schema documented with an example for the formal MDP spec (Overleaf design doc)
3. A README that lets Swapnil clone, `uv sync`, and run the same experiment
4. Config schema is stable across 2+ different dataset schemas and 2+ MDP designs (stretch)
5. Public dataset feasibility report: evaluate All of Us and UK Biobank for simulator training

Papers run *on* the framework later. Phase 2 (real-data validation) follows the foundation phase — core deliverable is working infrastructure.

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
- Baselines: Thompson Sampling
- Deep RL (stretch): DQN, PPO
- All share common interface
- Minimise dependencies — implement from scratch where reasonable

*Subphase 1E: Experiment Runner & Results*
- Config → train → compare → report
- Statistical comparison, visualisation, export
- Reproducibility (seeds, config snapshots)

**PHASE 2: REAL-DATA VALIDATION**  

- Integrate real wearable datasets (HeartSteps first, then All of Us, UK Biobank)
- Calibrate user simulator against observed behavioural responses
- Benchmark agents on real data distributions

**FUTURE (Post–Phase 2): LLM SIMULATION (Stretch)**  

- LLM-as-user architecture (prompt-driven persona simulation)
- Calibration against real wearable data distributions
- Statistical comparison: LLM-simulated vs real user behaviour

---

## MDP (formalised in `initial_design.tex`)

**State (14 variables, see initial_design.tex appendix):**
`steps_t`, `hr_t`, `sleep_hours_t`, `sedentary_min_t`, `time_of_day_t`,
`day_of_week_t`, `goal_progress_t`, `burden_t`, `a_{t-1}`,
`response_{t-1}`, `body_measure_k`, `age`, `gender`, `baseline_activity`

**Actions (discrete, configurable):** no message, motivational prompt, walking suggestion, goal reminder, recovery suggestion, progress feedback. Each action carries `burden_penalty` and `burden_penalty` (zero on no-op).

**Reward:** `R_t = α·Δsteps - β·burden_penalty_{a_t} + λ·goal_progress` (immediate) + `η·BM_improvement` (delayed body measure, every 21 days)

**Discount factor:** γ ∈ [0.9, 0.99] (open question — see `initial_design.tex` decision log)

**Dependencies:** Agent → MDP (Agent acts in the environment; the user simulation provides the reward and transition signals but the agent interface depends only on the MDP abstraction)

---

## Tech Stack

- **Language:** Python 3.11
- **Package manager:** `uv`
- **Config format:** YAML (confirmed with Mengyan)
- **Dependencies:** Minimal. No SB3 or Gymnasium unless proven necessary
- **Interface:** CLI + config files (web UI = stretch goal, not necessary at this stage)
- **Package:** Repo-based now; installable package if needed later (confirmed)

---

## Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Config format: YAML | Confirmed |
| 2 | Interface: CLI + config files; web UI = stretch | Confirmed |
| 3 | Language and package manager: Python 3.11 + uv | Confirmed |
| 4 | Phase 1 data: synthetic (real data requires 4–8 week applications while arrangements are underway) | Open |
| 5 | Baseline agent: Thompson Sampling | Confirmed |
| 6 | Decision frequency: daily epochs | Confirmed |
| 7 | Reward structure: multi-timescale (immediate + delayed) | Confirmed |
| 8 | User archetypes: goal-driven, social responder, resistant, stable maintainer | Confirmed |
| 9 | Component wiring: ABC + registry pattern | Confirmed |
| 10 | Dependencies: minimal — no Gymnasium, no SB3 | Confirmed |
| 11 | Distribution: repo-based now, package if needed later | Confirmed |
| 12 | MDP formalisation | Awaiting supervisor approval (Swapnil) |
| 13 | Phase 2 dataset priority: HeartSteps first, then All of Us, UK Biobank | Open |
| 14 | Experiment output format: CSV, terminal table, JSON, or summary plots | Open |
| 15 | Success metrics: regret, reward, adherence | Open |
| 16 | Action penalties (burden_penalty, burden_penalty) per archetype | Open |
| 17 | Discount factor γ ∈ [0.9, 0.99]: optimal range for PA interventions | Open |
| 18 | Decision epoch frequency: daily (base library of policies; per-user refinement within constraints). Hourly is an alternative. | Open |
| 19 | Merge reward penalty and burden penalty: currently separate (different roles: direct reward vs. state-accumulator). Merging is possible if simpler. | Open |
| 20 | Burden model: linear accumulator max(0, burden_t + penalty) with threshold decay. StepCountJITAI uses bounded [0,1] habituation. | Open |
| 21 | Activity metric for immediate reward: steps (default). Active minutes, METs, or composites are configurable alternatives. | Open |
| 22 | Sparse vs decaying delayed reward: 3-week sparse (current). Decaying formulation (η · BM · γ^{3k-t}) may improve credit assignment. | Open |

---

## Public Datasets (Feasibility Complete ✅)

Two datasets evaluated for simulator training. Full study: [`sources/data_sources.md`](sources/data_sources.md).

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

**Supplementary finding:** HeartSteps V1/V2 micro-randomized trials contain the only available *intervention response* data. These are smaller (50–100 participants) but directly calibrate the user simulation (1C). See [`sources/additional_data_sources.md`](sources/additional_data_sources.md).

## Logging & Error Handling

See canonical setup in [`code_design.md`](code_design.md#logging--error-handling-canonical).

Project-wide observability concerns:

- The codebase plan defines six subphase logger namespaces under
  `rl_health_interventions.{data,transitions,rewards,agents,simulation,experiment}`.
  The runner namespace is the parent.
- Phase 1 ships with stdlib `logging` only; no `structlog` or other third-party
  logging dependency.
- All CLI flags (`--verbose`, `--quiet`, `--log-file`) are exposed on the
  experiment runner entry point only — not on individual subphase CLIs.
- Per-episode exception isolation is a top-level framework requirement, not
  per-subphase.

## Literature Review Reference

See `initial_design.tex` §2 for the full literature review.

5 major gaps identified (experiments *on* the framework, not *in* it):
1. No LLM simulation validated for wearable health data
2. No delayed-feedback bandit for 3-week clinical measures
3. No non-stationary adaptation for PA interventions
4. No standardised MDP/benchmark for PA JITAI RL
5. No offline RL for PA interventions with real wearable data
