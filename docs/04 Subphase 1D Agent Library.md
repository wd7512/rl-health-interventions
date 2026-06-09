# Subphase 1D: Config-Driven Agent Library

**Status:** `[ ]` Not started

**Dependencies:** 1B environment interface (defined, not necessarily complete)

**Parallelises with:** 1A, 1C (after 1B interface agreed)

---

## Gate Checklist (MVP)

- [ ] Agent interface defined: `Agent.select_action(state) → action` and `Agent.update(experience)`
- [ ] Thompson Sampling agent implemented from scratch (no external RL lib)
- [ ] Agent reads hyperparameters from config (e.g. prior parameters, exploration rate)
- [ ] All agents share a common `Agent` base class
- [ ] `uv run pytest` passes for all 1D tests
- [ ] `uv run ruff check` and `uv run ty check` pass

---

## Stretch Goals (Not in Gate)

- [ ] ε-greedy baseline
- [ ] LinUCB baseline
- [ ] DQN (from scratch or minimal impl)
- [ ] Double DQN
- [ ] PPO
- [ ] CQL (offline)
- [ ] IQL (offline)

---

## TDD Checklist

- [ ] Write test for agent interface contract *before* implementing
- [ ] Write test for Thompson Sampling producing different actions for different contexts *before* implementing
- [ ] Write test for agent learning (regret decreases over time) *before* optimising

---

## Key Interfaces

### `AgentConfig`
```python
class AgentConfig(BaseModel):
    type: str  # e.g. "thompson_sampling"
    hyperparameters: dict[str, Any]
```

### `Agent`
```python
class Agent(ABC):
    @abstractmethod
    def select_action(self, state: State) -> Action

    @abstractmethod
    def update(self, state: State, action: Action, reward: float, next_state: State) -> None
```

---

## Blocking Risks

- **Thompson Sampling simplicity:** For the MVP, this is a well-understood algorithm. Low risk.
- **State space dimensionality:** If the state space is large, TS may need dimensionality reduction or feature selection. Config should allow specifying which state variables the agent uses.
- **Minimising dependencies:** TS can be implemented with numpy only. No external RL lib needed.
