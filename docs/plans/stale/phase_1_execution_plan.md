# Phase 1 — Execution Plan

**Status:** `[ ]` Planning phase

---

## Dependency Graph

```
 Track A                     Track B                     Track C
 ───────                     ───────                     ───────
 1A Data Layer [ ]           1B MDP Environment [ ]      Dataset Exploration [✅]
      │                            │                           │
      │                            ├──→ 1D Agent Lib [ ]       │
      │                            │   (interface first,       │
      │                            │    MVP: TS only)           │
      └────────┬───────────────────┘                           │
               ↓                                                ↓
         1C User Simulation [ ]  ←─────────────────────────────┘
               |               (gate: synth profiles + dataset report)
               ↓
         1E Experiment Runner [ ]
```

### Parallel Tracks

| Track | Subphases | Requires |
|-------|-----------|----------|
| A | 1A → | None |
| B | 1B → 1D | None → 1B interface |
| C | Dataset exploration → feeds 1C | None |
| Merge | 1C (after 1A + 1B + exploration) | 1A, 1B, exploration done |
| Final | 1E | 1C + 1D |

---

## Gates Summary

| Subphase | Gate | Blocks | Parallel With |
|----------|------|--------|---------------|
| 1A | Ingest + feature pipeline + synthetic data work, tested | 1C | 1B, 1D, Exploration |
| 1B | Environment interface defined + config-driven, tested | 1C | 1A, 1D (interface only), Exploration |
| Dataset Exploration | ✅ DONE — see `sources/data_sources.md` | 1C gate | 1A, 1B |
| Additional Data Sources | Survey of JITAI trials (HeartSteps V1/V2) and accessible benchmarks — see `sources/additional_data_sources.md` | 1C calibration | 1A, 1B |
| 1C | Rule-based user profiles + dataset report exist, tested | 1E | 1D |
| 1D | Thompson Sampling agent implemented, configured from config, tested | 1E | 1A, 1B (interface only) |
| 1E | End-to-end experiment runs, results output, tested | — | — |

---

## Subphase Files

- [subphase_1a_data_layer.md](subphase_1a_data_layer.md)
- [subphase_1b_mdp_environment.md](subphase_1b_mdp_environment.md)
- [subphase_1c_user_simulation.md](subphase_1c_user_simulation.md)
- [subphase_1d_agent_library.md](subphase_1d_agent_library.md)
- [subphase_1e_experiment_runner.md](subphase_1e_experiment_runner.md)

---

## Risks

- **Data access:** Both datasets require institutional applications (4-8 weeks). Phase 1 uses synthetic data; real data integration is Phase 2. Exploration report complete (`sources/data_sources.md`).
- **HeartSteps access:** HeartSteps V1/V2 (see `sources/additional_data_sources.md`) contains the only available *intervention response* data. Start access request early to avoid blocking 1C calibration.
- **Scope creep on 1D:** Deep RL agents explicitly excluded from MVP gate. TS only.
- **MDP confirmation:** Awaiting Swapnil sign-off on MDP spec. If he wants major changes, 1B interface may need rework. Note: `initial_design.tex` splits MDP across Config/transition/reward responsibilities; code mirrors this via `transitions/` + `rewards/` sub-packages — not a single monolithic module.
- **Overlap tracking:** Parallel tracks need explicit integration points at 1C and 1E.
- **Test organization:** Unit tests organised by component module (transitions/, rewards/, etc.). Gate coverage measured by module, not by test directory structure.

---

## Observability

The framework's observability contract is defined in
[`code_design.md`](code_design.md#logging--error-handling-canonical).
Each subphase doc specifies what's unique to that subphase; the runner
implementation owns the CLI flags, per-episode exception isolation, the
progress heartbeat, and the structured trace emission.

Phase 1 ships with stdlib `logging` only — no third-party logging
dependency. Adding `structlog` is a future consideration (deferred to
Phase 2 once we know whether the JSON formatter on top of stdlib logging
is sufficient).

