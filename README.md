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

Phase 1 delivers: config-driven data layer, MDP environment, rule-based user simulation, RL agent library, and experiment runner. Phase 2 (stretch) adds LLM-based user simulation.

See `docs/01 Codebase Plan.md` for full architecture.

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
│  □ DataConfig schema        │   │  □ MDPConfig schema               │   │  □ Investigate All of Us│
│  □ Polars lazy reader       │   │  □ TransitionModel ABC +          │   │  □ Investigate UK Biobk │
│  □ FeaturePipeline          │   │    RuleBasedTransition            │   │  □ Write report → docs/ │
│  □ Dataset + StateView      │   │  □ RewardHandler ABC +            │   └──────────┬─────────────┘
│    .from_dataset() bridge   │   │    CompoundReward                 │              │
│  □ SyntheticDataGenerator   │   │  □ FatigueTracker                 │              │
└──────────┬──────────────────┘   │  □ Environment step/reset         │              │
           │                      │  □ Multi-timescale reward         │              │
           │                      └──────────┬────────────────────────┘              │
           │                                 │                                        │
           │                                 │ (interface only)                       │
           │                                 ├──────────┐                             │
           │                                 │          │                             │
           │                                 ▼          ▼                             │
           │                          ┌─ 1D: Agent Lib ────┐                          │
           │                          │  □ Agent ABC        │                          │
           │                          │  □ ThompsonSampling │                          │
           │                          └─────────────────────┘                          │
           │                                                                          │
           └────────────────────────────┬─────────────────────────────────────────────┘
                                        ▼
                         ┌─ 1C: User Simulation ────────────────────────┐
                         │  □ UserProfile schema + 4 archetypes        │
                         │  □ ResponseModel ABC                        │
                         │  □ Backlash / fatigue mechanism              │
                         └──────────────────────┬───────────────────────┘
                                                ▼
                         ┌─ 1E: Experiment Runner ──────────────────────┐
                         │  □ ExperimentConfig + full config parsing    │
                         │  □ CLI → ExperimentFactory wiring + e2e test│
                         │  □ Results table + reproducibility          │
                         └──────────────────────────────────────────────┘
```

Each `□` is a GitHub issue. Milestones gate on all their issues closed and CI passing.
See `docs/phase-1-execution-plan.md` and `docs/06 Code Design.md` for detailed specifications.
