# rl-health-interventions

A configurable simulation framework for testing RL-driven health interventions on wearable device data.

## Quickstart

```bash
uv sync --dev
uv run rl-health-interventions
```

## Development Commands

```bash
uv run ruff format
uv run ruff check
uv run ty check
uv run pytest
```

## Project Structure

```
src/rl_health_interventions/   # package source
docs/                          # documentation and plans
tests/                         # test suite
```

## Overview

Config-first design. Researchers define their dataset schema, MDP, agents, and experiments via config files — no source code changes needed.

Phase 1 delivers: config-driven data layer, MDP environment, rule-based user simulation, RL agent library, and experiment runner. Phase 2 validates the framework against real datasets (HeartSteps V1/V2, All of Us Fitbit Dataset, UK Biobank Accelerometer Dataset), calibrating the user simulator and grounding MDP dynamics on observed behavioural responses. Beyond Phase 2, stretch goals include LLM-based user simulation.

See `docs/code/codebase_plan.md` and `docs/code/code_design.md` for full architecture.

## Milestones & Issues

```
┌─────────────────────────────────────────────────────────────────────┐
│              PHASE 1 — ISSUE / MILESTONE MAP                        │
└─────────────────────────────────────────────────────────────────────┘

  ── parallel    ── blocks    ──> feeds into


┌─ Foundation ─────────────────────────────────────────────────────────┐
│  □ Add deps (pydantic, numpy, polars)                                │
│  □ Create module dirs + REGISTRY dicts                               │
│  □ ExperimentFactory skeleton + 3-layer validation stubs             │
│  □ Test directory structure                                          │
└──────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─ 1A: Data Layer ───────────┐   ┌─ 1B: MDP Environment ─────────────┐   ┌─ Dataset Exploration ────┐
│  □ DataConfig schema        │   │  □ MDPConfig schema               │   │  ☑ Investigate All of Us│
│  □ Polars lazy reader       │   │  □ TransitionModel ABC +          │   │  ☑ Investigate UK Biobank│
│  □ FeaturePipeline          │   │    RuleBasedTransition            │   │  ☑ Write report → sources/data_sources.md│
│  □ Dataset + StateView      │   │  □ RewardHandler ABC +            │   └─────────────────────────┘
│    .from_dataset() bridge   │   │    CompoundReward                 │
│  □ SyntheticDataGenerator   │   │  □ FatigueTracker                 │
└─────────────────────────────┘   │  □ Environment step/reset         │
                                   │  □ Multi-timescale reward         │
                                   └──────────┬────────────────────────┘
                                              │
                                              │ (interface only)
                                              ├──────────┐
                                              │          │
                                              ▼          ▼
                                       ┌─ 1D: Agent Lib ────┐
                                       │  □ Agent ABC        │
                                       │  □ ThompsonSampling │
                                       └─────────────────────┘
                                              │
                                              │
                                              ▼
                            ┌─ 1C: User Simulation ────────────────────┐
                            │  □ UserProfile schema + 4 archetypes    │
                            │  □ ResponseModel ABC                    │
                            │  □ Backlash / fatigue mechanism          │
                            └─────────────────┬───────────────────────┘
                                              ▼
                            ┌─ 1E: Experiment Runner ──────────────────┐
                            │  □ ExperimentConfig + full config parsing│
                            │  □ CLI → ExperimentFactory wiring + e2e  │
                            │  □ Results table + reproducibility      │
                            └──────────────────────────────────────────┘
```

Each `□` is a GitHub issue. Milestones gate on all their issues closed and CI passing.
See `docs/code/phase_1_execution_plan.md` and `docs/code/code_design.md` for detailed specifications.

See also `docs/sources/additional_data_sources.md` for a survey of JITAI trial datasets (HeartSteps V1/V2) that provide intervention response data for calibrating the user simulation (1C) and agent benchmarks (1D).
