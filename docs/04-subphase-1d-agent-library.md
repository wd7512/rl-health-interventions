# Subphase 1D: Config-Driven Agent Library

**Status:** `[ ]` Not started

**Dependencies:** 1B environment interface (defined, not necessarily complete)

**Parallelises with:** 1A, 1C (after 1B interface agreed)

---

## Gate Checklist (MVP)

- [ ] Agent interface defined: `Agent.select_action(state: StateView) → int` and `Agent.update(state, action, reward, next_state) → None`
- [ ] Thompson Sampling agent implemented from scratch (no external RL lib)
- [ ] Agent reads hyperparameters from config (e.g. prior parameters, exploration rate)
- [ ] All agents share a common `Agent` base class
- [ ] `uv run pytest` passes for all 1D tests
- [ ] `uv run ruff check` and `uv run ty check` pass

---

## Stretch Goals (Not in Gate)

- [ ] epsilon-greedy baseline
- [ ] LinUCB baseline
- [ ] DQN (from scratch or minimal impl)
- [ ] Double DQN
- [ ] PPO
- [ ] CQL (offline)
- [ ] IQL (offline)

---

## TDD Checklist

- [ ] Write test for agent interface contract (select_action returns valid action idx, update accepts any state) *before* implementing
- [ ] Write test for Thompson Sampling producing different action distributions for different state contexts *before* implementing
- [ ] Write test for agent learning: regret decreases over time on a known-optimal bandit problem *before* optimising

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
    def select_action(self, state: StateView) -> int

    @abstractmethod
    def update(self, state: StateView, action: int, reward: float, next_state: StateView) -> None
```

---

## Blocking Risks

- **Thompson Sampling simplicity:** For the MVP, this is a well-understood algorithm. Low risk.
- **State space dimensionality:** If the state space is large, TS may need dimensionality reduction or feature selection. Config should allow specifying which state variables the agent uses.
- **Minimising dependencies:** TS can be implemented with numpy only. No external RL lib needed.

---

## Logging & Error Handling

See canonical setup in [`06 Code Design.md`](06%20Code%20Design.md#logging--error-handling-canonical).

Subphase-specific concerns for 1D (agent library):

- **Training convergence warnings:** If an agent's running mean reward
  decreases for 50 consecutive episodes, log WARNING with the agent id
  and current mean. Indicates a bug (e.g., wrong sign on reward) or a
  non-stationarity the agent can't handle.
- **Numerical instability:** Any NaN or Inf in agent parameters (posterior
  for TS, weights for neural agents) is logged at ERROR and the agent is
  reset to its prior. Never let NaN propagate into the next action.
- **Posterior diagnostics (TS):** Every 100 episodes, log INFO with the
  posterior mean and variance for the top 5 action arms. Useful for
  diagnosing under- or over-exploration.
- **Policy distribution (epsilon-greedy):** Every 100 episodes, log INFO
  with the action selection distribution (counts per action id). Should
  approach the optimal distribution as training progresses.
- **Update failures:** If `Agent.update()` raises, log ERROR with the
  offending `(state, action, reward, next_state)` and skip the update.
  One bad update must not crash the training loop.

Related 1D tests:
- `tests/unit/agents/test_ts_convergence.py` — posterior concentrates on
  the optimal arm after enough updates.
- `tests/unit/agents/test_nan_reset.py` — NaN in posterior triggers reset
  to prior.
- `tests/unit/agents/test_update_failure_isolation.py` — update raising
  is caught, next update proceeds.
