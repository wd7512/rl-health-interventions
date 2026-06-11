1. **CRITICAL**: `docs/04 Subphase 1D Agent Library.md` defines `Agent.update()` with three different signatures. The Gate Checklist says `update(experience)` (single arg), the Key Interfaces section says `update(self, state, action, reward, next_state)` (four args), and `docs/06 Code Design.md` says `update(self, state: StateView, action: int, reward: float, next_state: StateView)` (typed four args). Tests written against any one will be incompatible with the others, and no interface can satisfy all three.

2. **MEDIUM**: The TDD checklists across all subphase docs are too vague to be actionable. "Write test for synthetic data shape + statistical properties", "Write test for each behavioural archetype producing expected response direction", "Write test for config schema parsing" — none specify concrete criteria, edge cases, or thresholds. These will produce tests that pass by construction or miss critical behaviors entirely.

3. **MEDIUM**: No mapping exists between the test organization in `docs/06 Code Design.md` (by module: `transitions/`, `rewards/`, `agents/`, `simulation/`) and the gate test references (by subphase: 1A, 1B, etc.). This creates ambiguity about where subphase tests live, risking duplication or orphaned test files.

4. **LOW**: The REGISTRY-based plugin discovery pattern (central to the architecture) has no specified test coverage. No test verifies that adding an entry to `__init__.py` makes a component discoverable by the factory or that a missing REGISTRY entry produces a clear error.

5. **LOW**: The factory's Layer 3 dummy step (explicitly listed as a validation layer in Code Design) has no corresponding test requirement. No test specifies that wiring errors should be caught here or that a valid config's dummy step must succeed.
