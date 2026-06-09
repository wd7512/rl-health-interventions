# Phase 1 — Execution Plan

**Status:** `[ ]` Planning phase

---

## Dependency Graph

```
 Track A                     Track B                     Track C
 ───────                     ───────                     ───────
 1A Data Layer [ ]           1B MDP Environment [ ]      Dataset Exploration [ ]
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
| Dataset Exploration | Report evaluating All of Us + UK Biobank for sim training | 1C gate | 1A, 1B |
| 1C | Rule-based user profiles + dataset report exist, tested | 1E | 1D |
| 1D | Thompson Sampling agent implemented, configured from config, tested | 1E | 1A, 1B (interface only) |
| 1E | End-to-end experiment runs, results output, tested | — | — |

---

## Subphase Files

- [01-subphase-1a-data-layer.md](01-subphase-1a-data-layer.md)
- [02-subphase-1b-mdp-environment.md](02-subphase-1b-mdp-environment.md)
- [03-subphase-1c-user-simulation.md](03-subphase-1c-user-simulation.md)
- [04-subphase-1d-agent-library.md](04-subphase-1d-agent-library.md)
- [05-subphase-1e-experiment-runner.md](05-subphase-1e-experiment-runner.md)

---

## Risks

- **Data access:** All of Us / UK Biobank access may take weeks. Exploration report can proceed with public documentation.
- **Scope creep on 1D:** Deep RL agents explicitly excluded from MVP gate. TS only.
- **MDP confirmation:** Awaiting Swapnil sign-off on MDP spec. If he wants major changes, 1B interface may need rework.
- **Overlap tracking:** Parallel tracks need explicit integration points at 1C and 1E.
- **Test organization:** Unit tests organised by component module (transitions/, rewards/, etc.). Gate coverage measured by module, not by test directory structure.
