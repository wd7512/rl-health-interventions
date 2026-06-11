## Logic Review — docs/ PR (7 new .md files)

**HIGH**

1. **`ResponseModel.response()` return type mismatch** — Subphase 1C (`03 ...`) declares `→ float`, Code Design (`06 ...`) declares `→ StateView`. One of these is wrong. If the response model modifies the user state it must return a state, but the 1C interface returns a scalar (behavioural response strength). Integration of 1C into the step loop will fail.

2. **`Dataset` → `StateView` bridge missing** — 1A produces a `Dataset` with `users`, `timestamps`, `features`. The MDP environment and all downstream components consume `StateView`/`State`. No document explains how one becomes the other. This is an architectural gap that will surface as a wiring failure at the 1E integration gate.

3. **Scheduling contradiction: 1B blocks 1D?** — Gate Summary table (Phase 1 – Execution Plan) says 1B blocks 1D (full completion). Subphase 1D says dependency is only "environment interface (defined, not necessarily complete)". The dependency graph annotation `(interface first` supports the looser dependency. The table contradicts the docs and the graph.

**MEDIUM**

4. **`State` vs `StateView`** — Subphase docs (1B, 1C, 1D, 1E) all use `State`. Code Design renames it to `StateView`. No cross-reference, no aliasing, no migration note.

5. **Status inconsistency** — Execution Plan header says `[~] In progress`, graph shows `1A [~]`, but Subphase 1A doc body says `[ ] Not started`. Minor but confuses what state the project is actually in.

6. **Action type drift** — Subphase interfaces use `Action` (opaque). Code Design uses `int`. If `Action` is meant to be an int alias this is fine, but it's never documented, and the two conventions will collide in factory wiring.

**LOW**

7. **1D "Parallelises with: 1C"** — Technically possible (1D starts after 1B interface, 1C starts after 1A+1B), but the dependency graph shows 1C downstream of both 1A and 1B, so the overlap window is narrow and the claim may mislead scheduling.
