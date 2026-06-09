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
class MDPConfig(BaseModel):
    state_variables: list[str]
    actions: list[ActionSpec]
    reward_weights: RewardWeights
    transition_model: str  # e.g. "rule_based"
    gamma: float
    body_measure_interval: int  # epochs
```

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

---

## Blocking Risks

- **MDP confirmation pending:** If Swapnil wants significant changes to the MDP structure, the environment interface may need rework. Mitigation: design the config schema around the google doc MDP and flag that deviations are config changes, not code changes.
- **Gymnasium dependency:** Not needed yet. Keep it minimal.
