# Subphase 1B: Config-Driven MDP Environment

**Status:** `[ ]` Not started

**Dependencies:** None

**Parallelises with:** 1A, Dataset Exploration

---

## Gate Checklist

- [ ] MDP config schema defines states, actions, transitions, and rewards from config
- [ ] Environment exposes a clean step API: `env.step(state, action) → next_state, reward, done`
- [ ] Multi-timescale reward works: immediate step reward + delayed body measure at configurable intervals
- [ ] Rule-based transition model is pluggable (swap implementations via config)
- [ ] `uv run pytest` passes for all 1B tests
- [ ] `uv run ruff check` and `uv run ty check` pass

---

## TDD Checklist

- [ ] Write test for config-to-environment instantiation (valid config builds, invalid config raises) *before* implementing
- [ ] Write test for step API contract (returns (StateView, float, bool), state fields match config) *before* implementing transitions
- [ ] Write test for delayed reward at correct epoch (body measure reward non-zero only at interval boundary) *before* implementing reward accumulator

---

## Key Interfaces

### `MDPConfig`
```python
class ActionSpec(BaseModel):
    label: str
    reward_penalty: float = Field(ge=0.0)
    burden_penalty: float = Field(ge=0.0)


class MDPConfig(BaseModel):
    state_variables: list[str]
    actions: list[ActionSpec]
    reward_weights: RewardWeights
    transition_model: str  # e.g. "rule_based"
    gamma: float = Field(ge=0.9, le=0.99)
    body_measure_interval: int  # epochs
```

The `gamma` range `[0.9, 0.99]` is a framework constraint — the MDP is designed
for long-term behaviour change, not myopic optimisation. Reward weights
(α, β, λ, η) follow the same bounded Pydantic convention.

### `Environment`
```python
class Environment:
    def step(self, state: StateView, action: int) -> tuple[StateView, float, bool]
    def reset(self) -> StateView
```

### `TransitionModel`
```python
class TransitionModel(ABC):
    @abstractmethod
    def transition(self, state: StateView, action: int, user_profile: UserProfile) -> StateView
```

Note: `initial_design.tex` groups both `TransitionModel` and `RewardHandler`
under the MDP module. The `RewardHandler` ABC lives in `rewards/_base.py`
(see `code_design.md`); the environment's `step()` wires both together.

---

## Blocking Risks

- **MDP confirmation pending:** If Swapnil wants significant changes to the MDP structure, the environment interface may need rework. Mitigation: design the config schema around the formal MDP (`initial_design.tex`) and flag that deviations are config changes, not code changes.
- **Gymnasium dependency:** Not needed yet. Keep it minimal.

---

## Logging & Error Handling

See canonical setup in [`code_design.md`](code_design.md#logging--error-handling-canonical).

Subphase-specific concerns for 1B (MDP environment):

- **DEBUG events (per step):** state hash, action id, reward value, next_state
  hash, done flag. Disabled by default — enable with `--verbose` for debugging
  a single episode, not for production runs.
- **INFO events:** "Episode N started (seed=S)" at episode boundary; "Episode
  N ended at step T with total reward R" at termination.
- **Invalid action handling:** `Environment.step()` validates the action
  against the registered action space. Out-of-range actions are logged at
  WARNING with the offending value and rejected (return current state, zero
  reward, `done=False`). Never raise on invalid action in a long-running
  experiment.
- **Transition model failures:** If a `TransitionModel.transition()` raises,
  the episode is caught at the runner level, logged at ERROR, and counted
  as a failed episode. The MDP itself does not catch.
- **Step counter overflow:** Episodes are typically bounded (e.g., 365 steps
  for 1 year of daily decisions). The environment must log a WARNING if the
  step count exceeds 2× the configured max — indicates a buggy policy loop.

Related 1B tests:
- `tests/unit/transitions/test_invalid_action.py` — out-of-range action
  returns current state without raising.
- `tests/integration/test_dummy_step.py` — full build + step + reset
  succeeds, all 5 dimensions of state evolve correctly.
- `tests/unit/transitions/test_step_overflow_warning.py` — exceeding
  2× max steps emits WARNING.
