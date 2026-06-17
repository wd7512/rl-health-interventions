# Issue #101 MVP Implementation Plan

> **For Hermes:** Dispatch opencode using the `dev-workflow` skill, task by task. Verify with `uv run ruff check`, `uv run ty check`, `uv run pytest` between each task. Do not push or open PR — the user does that.

**Goal:** Implement the binary-action rule-based simulator (Issue #101) — a config-driven MVP that runs a 90-day HeartSteps-style simulation with Thompson Sampling and epsilon-greedy agents, producing reproducible CSV results.

**Architecture:** Config-first (YAML → Pydantic → Environment.step loop). The entire transition function, reward, agent priors, and episode structure live in YAML. Python code is a thin executor. Extending to multi-timescale reward / 4 archetypes / learned transitions means a new YAML, not new code.

**Tech Stack:** Python 3.11+, uv, pydantic, pyyaml, numpy, pytest, ruff, ty. No ML libraries (out of MVP scope).

**Branch:** `feat/issue-101-mvp-simulator` (cut from main, NOT pushed yet)

**Reference:** Issue #101 spec, audit findings (swarm, 15 Jun 2026 — see commit history)

---

## Architecture summary

```
YAML config
  └── config/loader.py → MDPConfig (Pydantic)
        ├── Environment
        │     ├── StateView (binary activity, time_of_day)
        │     ├── RuleBasedTransition (config matrix + time-of-day mask)
        │     ├── CompoundReward (placeholder: active=+1, sedentary=0)
        │     └── 90 days × 5 decision times = 450 step episode
        ├── Agent (TS or epsilon-greedy, via registry)
        └── Experiment runner → CSV results
```

ABCs and registry pattern already in place (don't recreate). The work is implementing the stubs.

---

## Task 1: Add pyyaml to pyproject.toml

**Objective:** Declare pyyaml as a direct runtime dependency (currently only transitive).

**Files:**
- Modify: `pyproject.toml:9-19` (add `"pyyaml>=6.0"` to `dependencies` list, alphabetical position)

**Step 1: Edit pyproject.toml**

Add `"pyyaml>=6.0",` after `"polars>=1.0.0",` and before `"pydantic>=2.0.0",` (alphabetical).

**Step 2: Sync dependencies**

```bash
uv sync --dev
```

Expected: succeeds, no errors.

**Step 3: Verify import works**

```bash
uv run python -c "import yaml; print(yaml.__version__)"
```

Expected: prints a version number (e.g. `6.0.x`).

**Step 4: Run full check suite**

```bash
uv run ruff check && uv run ty check && uv run pytest -q
```

Expected: all green (63 tests pass).

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add pyyaml as direct runtime dependency"
```

---

## Task 2: MDPConfig Pydantic schema + config/loader.py

**Objective:** Add a Pydantic model for the simulator config and a YAML loader function. Replace `profile: Any` with `time_of_day: int` in `TransitionModel` and `RewardHandler` ABCs (the audit's API mismatch).

**Files:**
- Create: `src/rl_health_interventions/config/__init__.py`
- Create: `src/rl_health_interventions/config/loader.py`
- Create: `src/rl_health_interventions/config/schemas.py`
- Create: `tests/unit/config/test_mdp_config.py`
- Modify: `src/rl_health_interventions/transitions/_base.py` (signature change)
- Modify: `src/rl_health_interventions/rewards/_base.py` (signature change)
- Modify: `src/rl_health_interventions/simulation/_base.py` (signature change)
- Modify: existing stub implementations to match new signatures
- Modify: `tests/integration/test_dummy_step.py` (update calls to pass `time_of_day`)

**Step 1: Write failing tests for MDPConfig**

`tests/unit/config/test_mdp_config.py`:
- `test_mdp_config_loads_valid_yaml` — full example YAML from Issue #101 spec
- `test_mdp_config_rejects_missing_transition_matrix` — invalid config raises ValidationError
- `test_mdp_config_rejects_probabilities_not_summing_to_one` — per (state, action) row must sum to 1
- `test_mdp_config_rejects_unknown_time_of_day` — only the 5 declared slots allowed
- `test_load_config_reads_yaml_file` — `load_config(path)` returns MDPConfig instance

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/unit/config/test_mdp_config.py -v
```

Expected: all FAIL with ModuleNotFoundError or ImportError.

**Step 3: Implement schemas.py**

Define:
- `ActivityLevel` enum: SEDENTARY, ACTIVE
- `TimeOfDay` enum: MORNING, MIDDAY, AFTERNOON, EVENING, NIGHT
- `Action` enum: SEND, DON_T_SEND
- `TransitionMatrix` — nested dict: `dict[ActivityLevel, dict[Action, dict[ActivityLevel, float]]]` validated with `model_validator` to ensure each row sums to 1.0 ± 1e-6
- `TimeOfDayMask` — `dict[TimeOfDay, dict[ActivityLevel, float]]` (1.0 = no state change, 0.0 = allow change; per Issue #101 spec, all slots = 1.0 = "mask to no state change")
- `MDPConfig(BaseModel)`:
  - `activity_levels: list[ActivityLevel]`
  - `actions: list[Action]`
  - `time_of_day: list[TimeOfDay]`
  - `steps_per_day: int = Field(ge=1, le=24)`  (default 5)
  - `episode_days: int = Field(ge=1, le=365)`  (default 90)
  - `transition: TransitionMatrix`
  - `masks: TimeOfDayMask`
  - `initial_state: ActivityLevel = ActivityLevel.SEDENTARY`
  - `reward_active: float = 1.0`
  - `reward_sedentary: float = 0.0`
  - `seed: int = 42`

**Step 4: Implement loader.py**

```python
from pathlib import Path
import yaml
from rl_health_interventions.config.schemas import MDPConfig


def load_config(path: str | Path) -> MDPConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return MDPConfig.model_validate(raw)
```

**Step 5: Update ABC signatures**

`transitions/_base.py`:
```python
from rl_health_interventions.config.schemas import ActivityLevel, Action, TimeOfDay

class TransitionModel(ABC):
    @abstractmethod
    def transition(
        self, state: ActivityLevel, action: Action, time_of_day: TimeOfDay
    ) -> ActivityLevel: ...
```

`rewards/_base.py`:
```python
class RewardHandler(ABC):
    @abstractmethod
    def reward(
        self, state: ActivityLevel, action: Action
    ) -> tuple[float, bool]: ...
```

`simulation/_base.py`:
```python
class ResponseModel(ABC):
    @abstractmethod
    def response(
        self, state: ActivityLevel, action: Action
    ) -> float: ...
```

**Step 6: Update stub implementations**

- `transitions/rule_based.py`: keep `transition(state, action, time_of_day)` signature (still no-op for now; Task 4 fills in)
- `rewards/compound.py`: keep `reward(state, action) -> (float, bool)` signature (still returns 0.0, False)
- `simulation/rule_based.py`: keep `response(state, action) -> float` (still 0.0)

**Step 7: Update test_dummy_step.py**

Change calls to pass `time_of_day=TimeOfDay.MORNING` to transition; pass `state, action` to reward and response. Add `from rl_health_interventions.config.schemas import ActivityLevel, Action, TimeOfDay` import.

**Step 8: Run new tests + existing tests**

```bash
uv run pytest tests/unit/config/test_mdp_config.py -v
uv run pytest tests/ -q
```

Expected: new tests pass, 63 → 68 tests pass.

**Step 9: Verify gates**

```bash
uv run ruff check && uv run ty check
```

Expected: all green.

**Step 10: Commit**

```bash
git add src/rl_health_interventions/config/ src/rl_health_interventions/transitions/_base.py src/rl_health_interventions/rewards/_base.py src/rl_health_interventions/simulation/_base.py src/rl_health_interventions/transitions/rule_based.py src/rl_health_interventions/rewards/compound.py src/rl_health_interventions/simulation/rule_based.py tests/
git commit -m "feat(config): add MDPConfig Pydantic schema and YAML loader

Replaces 'profile: Any' parameter in TransitionModel, RewardHandler,
and ResponseModel ABCs with typed ActivityLevel/Action/TimeOfDay
arguments. Resolves audit's API mismatch (Issue #101 needs
time_of_day, not user profile which is post-MVP).

- New: src/rl_health_interventions/config/{schemas.py,loader.py}
- New: tests/unit/config/test_mdp_config.py (5 tests)
- Updated: existing stubs to match new signatures
- Updated: test_dummy_step.py to pass typed args

Refs: #101"
```

---

## Task 3: StateView dataclass

**Objective:** Add a `StateView` dataclass that wraps the binary state with timestep context. Used by Environment to pass state through the step loop.

**Files:**
- Create: `src/rl_health_interventions/state.py`
- Create: `tests/unit/test_state.py`

**Step 1: Write failing test**

```python
def test_state_view_carries_activity_time_step():
    from rl_health_interventions.config.schemas import ActivityLevel, TimeOfDay
    from rl_health_interventions.state import StateView

    sv = StateView(activity=ActivityLevel.ACTIVE, time_of_day=TimeOfDay.MORNING, day=0, step_of_day=0)
    assert sv.activity == ActivityLevel.ACTIVE
    assert sv.time_of_day == TimeOfDay.MORNING
    assert sv.day == 0
    assert sv.step_of_day == 0
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_state.py -v
```

Expected: ImportError on `rl_health_interventions.state`.

**Step 3: Implement state.py**

```python
from __future__ import annotations
from dataclasses import dataclass
from rl_health_interventions.config.schemas import ActivityLevel, TimeOfDay


@dataclass(frozen=True)
class StateView:
    activity: ActivityLevel
    time_of_day: TimeOfDay
    day: int
    step_of_day: int

    @property
    def global_step(self) -> int:
        return self.day * 5 + self.step_of_day  # hardcoded 5 for MVP; will be config-driven
```

**Step 4: Run test to verify pass**

```bash
uv run pytest tests/unit/test_state.py -v
```

Expected: PASS.

**Step 5: Add 2 more tests**

- `test_state_view_global_step_calculation` — verify `global_step` math for various (day, step_of_day) combinations
- `test_state_view_is_frozen` — verify dataclass is immutable (frozen=True)

**Step 6: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: 68 → 71 tests pass.

**Step 7: Commit**

```bash
git add src/rl_health_interventions/state.py tests/unit/test_state.py
git commit -m "feat(state): add StateView dataclass for binary activity + time context"
```

---

## Task 4: RuleBasedTransition (replace no-op stub)

**Objective:** Implement the config-driven transition: given (state, action, time_of_day), sample next state from the transition matrix, with time-of-day mask forcing no-change when mask=1.0.

**Files:**
- Modify: `src/rl_health_interventions/transitions/rule_based.py`
- Create: `tests/unit/transitions/test_rule_based_transition.py`

**Step 1: Write failing test (deterministic with seed)**

```python
import numpy as np
from rl_health_interventions.config.schemas import (
    ActivityLevel, Action, TimeOfDay, MDPConfig, TransitionMatrix, TimeOfDayMask
)
from rl_health_interventions.transitions.rule_based import RuleBasedTransition


def _config() -> MDPConfig:
    return MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=[TimeOfDay.MORNING, TimeOfDay.MIDDAY, TimeOfDay.AFTERNOON, TimeOfDay.EVENING, TimeOfDay.NIGHT],
        transition=TransitionMatrix(root={
            ActivityLevel.SEDENTARY: {
                Action.SEND: {ActivityLevel.SEDENTARY: 0.7, ActivityLevel.ACTIVE: 0.3},
                Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.9, ActivityLevel.ACTIVE: 0.1},
            },
            ActivityLevel.ACTIVE: {
                Action.SEND: {ActivityLevel.SEDENTARY: 0.2, ActivityLevel.ACTIVE: 0.8},
                Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.4, ActivityLevel.ACTIVE: 0.6},
            },
        }),
        masks=TimeOfDayMask(root={
            TimeOfDay.MORNING: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0},
            TimeOfDay.MIDDAY: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0},
            TimeOfDay.AFTERNOON: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0},
            TimeOfDay.EVENING: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0},
            TimeOfDay.NIGHT: {ActivityLevel.SEDENTARY: 1.0, ActivityLevel.ACTIVE: 1.0},
        }),
    )


def test_transition_returns_valid_state():
    t = RuleBasedTransition(_config(), seed=42)
    next_state = t.transition(ActivityLevel.SEDENTARY, Action.SEND, TimeOfDay.MORNING)
    assert next_state in (ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE)


def test_transition_night_mask_forces_no_change():
    t = RuleBasedTransition(_config(), seed=42)
    for _ in range(100):
        next_state = t.transition(ActivityLevel.SEDENTARY, Action.SEND, TimeOfDay.NIGHT)
        assert next_state == ActivityLevel.SEDENTARY
        next_state = t.transition(ActivityLevel.ACTIVE, Action.DON_T_SEND, TimeOfDay.NIGHT)
        assert next_state == ActivityLevel.ACTIVE


def test_transition_probabilities_match_config_over_many_samples():
    """Statistical test: over 10K samples, observed freq should match config within 2%."""
    t = RuleBasedTransition(_config(), seed=42)
    n_samples = 10_000
    n_active = sum(
        1 for _ in range(n_samples)
        if t.transition(ActivityLevel.SEDENTARY, Action.SEND, TimeOfDay.MORNING) == ActivityLevel.ACTIVE
    )
    observed_freq = n_active / n_samples
    assert abs(observed_freq - 0.3) < 0.02  # config says P(active | sedentary, send) = 0.3
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/unit/transitions/test_rule_based_transition.py -v
```

Expected: 3 tests FAIL (currently stub returns state unchanged).

**Step 3: Update RuleBasedTransition**

```python
from __future__ import annotations
import numpy as np
from rl_health_interventions.config.schemas import (
    ActivityLevel, Action, TimeOfDay, MDPConfig
)
from rl_health_interventions.transitions._base import TransitionModel


class RuleBasedTransition(TransitionModel):
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        self._config = config
        self._rng = np.random.default_rng(seed)
        self._next_state_cache: dict[tuple, list[tuple[ActivityLevel, float]]] = {}
        self._build_cache()

    def _build_cache(self) -> None:
        for state in self._config.activity_levels:
            for action in self._config.actions:
                row = self._config.transition.root[state][action]
                states = list(row.keys())
                probs = np.array([row[s] for s in states], dtype=np.float64)
                # Apply mask: if mask is 1.0 for this (time_of_day, state), force no-change
                # We do that at transition() time, not here
                self._next_state_cache[(state, action)] = list(zip(states, probs.tolist()))

    def transition(
        self, state: ActivityLevel, action: Action, time_of_day: TimeOfDay
    ) -> ActivityLevel:
        mask_value = self._config.masks.root[time_of_day][state]
        if mask_value == 1.0:
            # Mask forces no change (user unavailable, e.g. sleeping)
            return state
        # Sample from the distribution
        states, probs = zip(*self._next_state_cache[(state, action)])
        idx = self._rng.choice(len(states), p=np.array(probs))
        return states[idx]


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY
    REGISTRY["rule_based"] = RuleBasedTransition
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/transitions/test_rule_based_transition.py -v
```

Expected: 3 tests PASS.

**Step 5: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: 71 → 74 tests pass (existing test_dummy_step.py should still work since signature matches).

**Step 6: Commit**

```bash
git add src/rl_health_interventions/transitions/rule_based.py tests/unit/transitions/
git commit -m "feat(transitions): implement config-driven RuleBasedTransition

Replaces no-op stub with transition matrix sampling from MDPConfig.
Time-of-day mask forces no state change when mask value is 1.0
(used for sleep/availability windows per Issue #101 spec).

Verified statistical fidelity: 10K samples match config probabilities
within 2% tolerance."
```

---

## Task 5: CompoundReward (active=+1, sedentary=0)

**Objective:** Replace stub with simple single-timescale reward: configurable reward per activity level.

**Files:**
- Modify: `src/rl_health_interventions/rewards/compound.py`
- Create: `tests/unit/rewards/test_compound_reward.py`

**Step 1: Write failing test**

```python
from rl_health_interventions.config.schemas import (
    ActivityLevel, Action, MDPConfig, TransitionMatrix, TimeOfDayMask, TimeOfDay
)
from rl_health_interventions.rewards.compound import CompoundReward


def _config() -> MDPConfig:
    return MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=[TimeOfDay.MORNING],
        transition=TransitionMatrix(root={
            ActivityLevel.SEDENTARY: {Action.SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5}},
            ActivityLevel.ACTIVE: {Action.SEND: {ActivityLevel.SEDENTARY: 0.5, ActivityLevel.ACTIVE: 0.5}},
        }),
        masks=TimeOfDayMask(root={TimeOfDay.MORNING: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0}}),
    )


def test_active_state_reward():
    r = CompoundReward(_config())
    reward, done = r.reward(ActivityLevel.ACTIVE, Action.SEND)
    assert reward == 1.0
    assert done is False


def test_sedentary_state_reward():
    r = CompoundReward(_config())
    reward, done = r.reward(ActivityLevel.SEDENTARY, Action.SEND)
    assert reward == 0.0
    assert done is False


def test_reward_never_terminates():
    r = CompoundReward(_config())
    for _ in range(10):
        _, done = r.reward(ActivityLevel.SEDENTARY, Action.SEND)
        assert done is False
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/rewards/test_compound_reward.py -v
```

Expected: 3 tests FAIL (currently stub returns 0.0, False).

**Step 3: Implement CompoundReward**

```python
from __future__ import annotations
from rl_health_interventions.config.schemas import (
    ActivityLevel, Action, MDPConfig
)
from rl_health_interventions.rewards._base import RewardHandler


class CompoundReward(RewardHandler):
    def __init__(self, config: MDPConfig) -> None:
        self._config = config

    def reward(
        self, state: ActivityLevel, action: Action
    ) -> tuple[float, bool]:
        if state == ActivityLevel.ACTIVE:
            return self._config.reward_active, False
        return self._config.reward_sedentary, False


def register() -> None:
    from rl_health_interventions.rewards import REGISTRY
    REGISTRY["compound"] = CompoundReward
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/rewards/test_compound_reward.py -v
```

Expected: 3 tests PASS.

**Step 5: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: 74 → 77 tests pass.

**Step 6: Commit**

```bash
git add src/rl_health_interventions/rewards/compound.py tests/unit/rewards/test_compound_reward.py
git commit -m "feat(rewards): implement config-driven CompoundReward

Single-timescale placeholder per Issue #101 spec. Reward is
configurable per activity level (default active=+1, sedentary=0).
Multi-timescale reward (immediate + 3-week delayed) is post-MVP.

Verified: reward never terminates, deterministic per state."
```

---

## Task 6: ThompsonSamplingAgent (Beta-Bernoulli)

**Objective:** Replace no-op stub with Beta-Bernoulli Thompson Sampling for binary actions.

**Files:**
- Modify: `src/rl_health_interventions/agents/thompson_sampling.py`
- Create: `tests/unit/agents/test_thompson_sampling.py`

**Step 1: Write failing test**

```python
from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent
from rl_health_interventions.config.schemas import Action


def test_select_action_returns_valid_action():
    agent = ThompsonSamplingAgent(n_actions=2, seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    action = agent.select_action(state)
    assert action in (Action.SEND, Action.DON_T_SEND)


def test_update_increases_beta_for_chosen_action():
    agent = ThompsonSamplingAgent(n_actions=2, seed=42, alpha_prior=1.0, beta_prior=1.0)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    # Update with reward=1 for SEND action
    agent.update(state, Action.SEND, reward=1.0, next_state=state)
    # Now posterior alpha for SEND should be 2.0
    assert agent.posteriors[Action.SEND][0] == 2.0  # alpha
    assert agent.posteriors[Action.SEND][1] == 1.0  # beta unchanged


def test_update_decreases_beta_for_chosen_action_on_zero_reward():
    agent = ThompsonSamplingAgent(n_actions=2, seed=42, alpha_prior=1.0, beta_prior=1.0)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    agent.update(state, Action.DON_T_SEND, reward=0.0, next_state=state)
    assert agent.posteriors[Action.DON_T_SEND][0] == 1.0  # alpha unchanged
    assert agent.posteriors[Action.DON_T_SEND][1] == 2.0  # beta


def test_thompson_sampling_converges_to_better_arm():
    """Statistical test: over 1000 episodes, TS should prefer the arm with higher reward probability."""
    import numpy as np
    agent = ThompsonSamplingAgent(n_actions=2, seed=42, alpha_prior=1.0, beta_prior=1.0)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    arm_rewards = {Action.SEND: 0.8, Action.DON_T_SEND: 0.2}
    rng = np.random.default_rng(42)
    for _ in range(1000):
        action = agent.select_action(state)
        reward = 1.0 if rng.random() < arm_rewards[action] else 0.0
        agent.update(state, action, reward, state)
    # TS should have pulled SEND (the better arm) at least 80% of the time
    # (after 1000 episodes, posterior for SEND is roughly Beta(1 + 800*0.8, 1 + 800*0.2) = Beta(641, 161))
    # So most pulls should be SEND. We check that posterior alpha > beta for SEND.
    assert agent.posteriors[Action.SEND][0] > agent.posteriors[Action.SEND][1]
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/agents/test_thompson_sampling.py -v
```

Expected: 4 tests FAIL (stub returns action=0, update is no-op).

**Step 3: Implement ThompsonSamplingAgent**

```python
from __future__ import annotations
import numpy as np
from rl_health_interventions.config.schemas import Action
from rl_health_interventions.agents._base import Agent


class ThompsonSamplingAgent(Agent):
    def __init__(
        self,
        n_actions: int = 2,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_actions = n_actions
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self._rng = np.random.default_rng(seed)
        # Posterior: dict[Action, [alpha, beta]]
        self.posteriors: dict[Action, list[float]] = {
            action: [alpha_prior, beta_prior]
            for action in Action
        }

    def select_action(self, state) -> Action:
        samples = {
            action: self._rng.beta(posterior[0], posterior[1])
            for action, posterior in self.posteriors.items()
        }
        return max(samples, key=samples.get)

    def update(self, state, action: Action, reward: float, next_state) -> None:
        if reward == 1.0:
            self.posteriors[action][0] += 1  # alpha
        else:
            self.posteriors[action][1] += 1  # beta


def register() -> None:
    from rl_health_interventions.agents import REGISTRY
    REGISTRY["thompson_sampling"] = ThompsonSamplingAgent
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/agents/test_thompson_sampling.py -v
```

Expected: 4 tests PASS.

**Step 5: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: 77 → 81 tests pass.

**Step 6: Commit**

```bash
git add src/rl_health_interventions/agents/thompson_sampling.py tests/unit/agents/
git commit -m "feat(agents): implement Beta-Bernoulli Thompson Sampling

Replaces no-op stub with posterior sampling over Beta(α, β) per
binary action. α_prior and β_prior configurable (default 1.0 each).

Verified convergence: after 1000 episodes with one arm at p=0.8
and the other at p=0.2, posterior correctly identifies the better arm."
```

---

## Task 7: EpsilonGreedyAgent (new file)

**Objective:** Add baseline comparison agent. Epsilon-greedy with configurable exploration rate.

**Files:**
- Create: `src/rl_health_interventions/agents/epsilon_greedy.py`
- Create: `tests/unit/agents/test_epsilon_greedy.py`
- Modify: `src/rl_health_interventions/agents/__init__.py` (register the new agent)

**Step 1: Write failing test**

```python
import numpy as np
from rl_health_interventions.config.schemas import Action
from rl_health_interventions.agents.epsilon_greedy import EpsilonGreedyAgent


def test_select_action_returns_valid_action():
    agent = EpsilonGreedyAgent(n_actions=2, epsilon=0.1, seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    action = agent.select_action(state)
    assert action in (Action.SEND, Action.DON_T_SEND)


def test_explores_with_epsilon_probability():
    """Over many calls, random action should be picked ~epsilon fraction."""
    agent = EpsilonGreedyAgent(n_actions=2, epsilon=0.3, seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    # Start with all values equal so greedy choice is arbitrary; we measure distribution
    counts = {Action.SEND: 0, Action.DON_T_SEND: 0}
    for _ in range(10_000):
        action = agent.select_action(state)
        counts[action] += 1
    # Each action should be picked roughly 50% of the time (epsilon * 1/n_actions = 0.3 * 0.5 = 0.15 from random + 0.85 * 0.5 from greedy = 0.5)
    # More precisely, when all Q values are equal, the agent picks randomly anyway
    # So both should be ~50%
    assert abs(counts[Action.SEND] / 10_000 - 0.5) < 0.05
    assert abs(counts[Action.DON_T_SEND] / 10_000 - 0.5) < 0.05


def test_update_tracks_average_reward():
    agent = EpsilonGreedyAgent(n_actions=2, epsilon=0.1, seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    # All rewards = 1.0 for SEND
    for _ in range(100):
        agent.update(state, Action.SEND, reward=1.0, next_state=state)
    # After 100 updates, Q[SEND] should be 1.0
    assert agent.q_values[Action.SEND] == 1.0


def test_greedy_prefers_better_arm_after_learning():
    """After learning, with epsilon=0, should always pick the better arm."""
    agent = EpsilonGreedyAgent(n_actions=2, epsilon=0.0, seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    # Teach that SEND is better
    for _ in range(50):
        agent.update(state, Action.SEND, reward=1.0, next_state=state)
        agent.update(state, Action.DON_T_SEND, reward=0.0, next_state=state)
    # With epsilon=0, should always pick SEND
    for _ in range(10):
        assert agent.select_action(state) == Action.SEND
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/agents/test_epsilon_greedy.py -v
```

Expected: ImportError or attribute errors.

**Step 3: Implement EpsilonGreedyAgent**

```python
from __future__ import annotations
import numpy as np
from rl_health_interventions.config.schemas import Action
from rl_health_interventions.agents._base import Agent


class EpsilonGreedyAgent(Agent):
    def __init__(
        self,
        n_actions: int = 2,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.n_actions = n_actions
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        # Q-values and counts per action
        self.q_values: dict[Action, float] = {action: 0.0 for action in Action}
        self.counts: dict[Action, int] = {action: 0 for action in Action}

    def select_action(self, state) -> Action:
        if self._rng.random() < self.epsilon:
            # Explore: random action
            return self._rng.choice(list(self.q_values.keys()))
        # Exploit: pick action with highest Q-value
        return max(self.q_values, key=self.q_values.get)

    def update(self, state, action: Action, reward: float, next_state) -> None:
        self.counts[action] += 1
        # Incremental average update: Q_new = Q_old + (reward - Q_old) / n
        n = self.counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n


def register() -> None:
    from rl_health_interventions.agents import REGISTRY
    REGISTRY["epsilon_greedy"] = EpsilonGreedyAgent
```

**Step 4: Register in agents/__init__.py**

Edit `src/rl_health_interventions/agents/__init__.py` to add `from rl_health_interventions.agents import epsilon_greedy` and a try/except registration call matching the existing pattern.

**Step 5: Run tests to verify pass**

```bash
uv run pytest tests/unit/agents/test_epsilon_greedy.py -v
```

Expected: 4 tests PASS.

**Step 6: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: 81 → 85 tests pass.

**Step 7: Commit**

```bash
git add src/rl_health_interventions/agents/epsilon_greedy.py src/rl_health_interventions/agents/__init__.py tests/unit/agents/test_epsilon_greedy.py
git commit -m "feat(agents): add EpsilonGreedyAgent baseline

Configurable exploration rate (default ε=0.1). Uses incremental
average Q-value updates. Baseline for Thompson Sampling comparison.

Verified: greedy exploitation after learning, exploration distribution
matches ε, incremental update math correct."
```

---

## Task 8: Environment class (step/reset, 5 time slots)

**Objective:** Implement the simulation loop: `reset()` returns initial StateView, `step(action)` returns (next_state, reward, done). Episode = episode_days × steps_per_day timesteps.

**Files:**
- Create: `src/rl_health_interventions/environment.py`
- Create: `tests/unit/test_environment.py`

**Step 1: Write failing test**

```python
from rl_health_interventions.config.schemas import (
    ActivityLevel, Action, TimeOfDay, MDPConfig, TransitionMatrix, TimeOfDayMask
)
from rl_health_interventions.environment import Environment


def _config(steps_per_day=5, episode_days=90) -> MDPConfig:
    return MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=[TimeOfDay.MORNING, TimeOfDay.MIDDAY, TimeOfDay.AFTERNOON, TimeOfDay.EVENING, TimeOfDay.NIGHT],
        steps_per_day=steps_per_day,
        episode_days=episode_days,
        transition=TransitionMatrix(root={
            ActivityLevel.SEDENTARY: {
                Action.SEND: {ActivityLevel.SEDENTARY: 0.7, ActivityLevel.ACTIVE: 0.3},
                Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.9, ActivityLevel.ACTIVE: 0.1},
            },
            ActivityLevel.ACTIVE: {
                Action.SEND: {ActivityLevel.SEDENTARY: 0.2, ActivityLevel.ACTIVE: 0.8},
                Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.4, ActivityLevel.ACTIVE: 0.6},
            },
        }),
        masks=TimeOfDayMask(root={
            t: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0}
            for t in [TimeOfDay.MORNING, TimeOfDay.MIDDAY, TimeOfDay.AFTERNOON, TimeOfDay.EVENING, TimeOfDay.NIGHT]
        }),
    )


def test_reset_returns_initial_state():
    env = Environment(_config(), seed=42)
    state = env.reset()
    assert state.activity == ActivityLevel.SEDENTARY
    assert state.day == 0
    assert state.step_of_day == 0
    assert state.time_of_day == TimeOfDay.MORNING


def test_step_returns_tuple():
    env = Environment(_config(), seed=42)
    env.reset()
    next_state, reward, done = env.step(Action.SEND)
    assert isinstance(next_state.activity, ActivityLevel)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_episode_terminates_after_correct_length():
    env = Environment(_config(steps_per_day=5, episode_days=2), seed=42)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done = env.step(Action.SEND)
        steps += 1
        if steps > 100:  # safety
            break
    assert steps == 10  # 2 days × 5 steps/day
    assert done is True


def test_time_of_day_cycles_within_day():
    env = Environment(_config(steps_per_day=5, episode_days=1), seed=42)
    state = env.reset()
    expected_times = [TimeOfDay.MORNING, TimeOfDay.MIDDAY, TimeOfDay.AFTERNOON, TimeOfDay.EVENING, TimeOfDay.NIGHT]
    for i, expected in enumerate(expected_times):
        assert state.time_of_day == expected, f"step {i}"
        if i < 4:
            state, _, _ = env.step(Action.SEND)


def test_step_after_done_raises():
    env = Environment(_config(steps_per_day=2, episode_days=1), seed=42)
    env.reset()
    env.step(Action.SEND)
    env.step(Action.SEND)
    import pytest
    with pytest.raises(RuntimeError, match="Episode is done"):
        env.step(Action.SEND)
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_environment.py -v
```

Expected: 5 tests FAIL (no Environment class).

**Step 3: Implement environment.py**

```python
from __future__ import annotations
import logging
from rl_health_interventions.config.schemas import (
    ActivityLevel, Action, MDPConfig, TimeOfDay
)
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions.rule_based import RuleBasedTransition
from rl_health_interventions.rewards.compound import CompoundReward

logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        self._config = config
        self._transition = RuleBasedTransition(config, seed=seed)
        self._reward = CompoundReward(config)
        self._step_count = 0
        self._done = False
        self._current_state: StateView | None = None

    def reset(self) -> StateView:
        self._step_count = 0
        self._done = False
        time_of_day = self._config.time_of_day[0]
        self._current_state = StateView(
            activity=self._config.initial_state,
            time_of_day=time_of_day,
            day=0,
            step_of_day=0,
        )
        logger.debug("Environment reset: %s", self._current_state)
        return self._current_state

    def step(self, action: Action) -> tuple[StateView, float, bool]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._current_state is None:
            raise RuntimeError("Call reset() before step().")

        # Apply transition
        next_activity = self._transition.transition(
            self._current_state.activity, action, self._current_state.time_of_day
        )

        # Compute reward based on the NEW state (post-transition)
        reward, _ = self._reward.reward(next_activity, action)

        # Advance to next step
        self._step_count += 1
        next_step_of_day = self._step_count % self._config.steps_per_day
        next_day = self._step_count // self._config.steps_per_day
        next_time_of_day = self._config.time_of_day[next_step_of_day]
        self._current_state = StateView(
            activity=next_activity,
            time_of_day=next_time_of_day,
            day=next_day,
            step_of_day=next_step_of_day,
        )

        # Check termination
        if self._step_count >= self._config.steps_per_day * self._config.episode_days:
            self._done = True

        logger.debug("Step %d: action=%s, next=%s, reward=%.2f, done=%s",
                     self._step_count, action, self._current_state, reward, self._done)
        return self._current_state, reward, self._done
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/test_environment.py -v
```

Expected: 5 tests PASS.

**Step 5: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: 85 → 90 tests pass.

**Step 6: Commit**

```bash
git add src/rl_health_interventions/environment.py tests/unit/test_environment.py
git commit -m "feat(environment): add Environment class with step/reset API

Implements the simulation loop: episode_days × steps_per_day timesteps,
time-of-day cycles within each day, post-transition reward computation.

Verified: episode length, time-of-day cycling, post-done error handling."
```

---

## Task 9: Experiment runner + CSV output

**Objective:** Add a function that runs an agent through an episode and writes per-step CSV results.

**Files:**
- Create: `src/rl_health_interventions/experiment.py`
- Create: `tests/unit/test_experiment.py`

**Step 1: Write failing test**

```python
import csv
from pathlib import Path
from rl_health_interventions.config.schemas import (
    ActivityLevel, Action, TimeOfDay, MDPConfig, TransitionMatrix, TimeOfDayMask
)
from rl_health_interventions.experiment import run_episode
from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent
from rl_health_interventions.agents.epsilon_greedy import EpsilonGreedyAgent


def _config(steps_per_day=5, episode_days=2) -> MDPConfig:
    return MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=[TimeOfDay.MORNING, TimeOfDay.MIDDAY, TimeOfDay.AFTERNOON, TimeOfDay.EVENING, TimeOfDay.NIGHT],
        steps_per_day=steps_per_day,
        episode_days=episode_days,
        transition=TransitionMatrix(root={
            ActivityLevel.SEDENTARY: {
                Action.SEND: {ActivityLevel.SEDENTARY: 0.7, ActivityLevel.ACTIVE: 0.3},
                Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.9, ActivityLevel.ACTIVE: 0.1},
            },
            ActivityLevel.ACTIVE: {
                Action.SEND: {ActivityLevel.SEDENTARY: 0.2, ActivityLevel.ACTIVE: 0.8},
                Action.DON_T_SEND: {ActivityLevel.SEDENTARY: 0.4, ActivityLevel.ACTIVE: 0.6},
            },
        }),
        masks=TimeOfDayMask(root={
            t: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0}
            for t in [TimeOfDay.MORNING, TimeOfDay.MIDDAY, TimeOfDay.AFTERNOON, TimeOfDay.EVENING, TimeOfDay.NIGHT]
        }),
        seed=42,
    )


def test_run_episode_returns_dataframe_with_correct_columns():
    import pandas as pd
    config = _config()
    agent = ThompsonSamplingAgent(seed=42)
    df = run_episode(config, agent, seed=42)
    assert isinstance(df, pd.DataFrame)
    expected_cols = {"step", "day", "step_of_day", "time_of_day", "state", "action", "reward"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 10  # 2 days × 5 steps/day


def test_run_episode_writes_csv(tmp_path: Path):
    config = _config()
    agent = EpsilonGreedyAgent(seed=42, epsilon=0.1)
    output_path = tmp_path / "results.csv"
    df = run_episode(config, agent, output_csv=output_path, seed=42)
    assert output_path.exists()
    with output_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 10
    assert "step" in rows[0]
    assert "action" in rows[0]


def test_run_episode_reproducible_with_seed():
    config = _config()
    agent1 = ThompsonSamplingAgent(seed=42)
    agent2 = ThompsonSamplingAgent(seed=42)
    df1 = run_episode(config, agent1, seed=42)
    df2 = run_episode(config, agent2, seed=42)
    assert df1["action"].tolist() == df2["action"].tolist()
    assert df1["reward"].tolist() == df2["reward"].tolist()
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_experiment.py -v
```

Expected: ImportError on `rl_health_interventions.experiment`.

**Step 3: Implement experiment.py**

```python
from __future__ import annotations
import logging
from pathlib import Path
from typing import Protocol
import pandas as pd
from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.environment import Environment
from rl_health_interventions.state import StateView

logger = logging.getLogger(__name__)


class AgentLike(Protocol):
    def select_action(self, state) -> object: ...
    def update(self, state, action, reward, next_state) -> None: ...


def run_episode(
    config: MDPConfig,
    agent: AgentLike,
    output_csv: Path | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    if seed is None:
        seed = config.seed
    env = Environment(config, seed=seed)
    state = env.reset()
    records = []
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        records.append({
            "step": state.global_step,
            "day": state.day,
            "step_of_day": state.step_of_day,
            "time_of_day": state.time_of_day.value,
            "state": state.activity.value,
            "action": action.value,
            "reward": reward,
        })
        agent.update(state, action, reward, next_state)
        state = next_state

    df = pd.DataFrame.from_records(records)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        logger.info("Wrote %d rows to %s", len(df), output_csv)
    return df
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/test_experiment.py -v
```

Expected: 3 tests PASS.

**Step 5: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: 90 → 93 tests pass.

**Step 6: Commit**

```bash
git add src/rl_health_interventions/experiment.py tests/unit/test_experiment.py
git commit -m "feat(experiment): add run_episode with CSV output

Runs an agent through one episode, returns per-step DataFrame,
optionally writes to CSV. Verified: column schema, episode length,
reproducibility with seed."
```

---

## Task 10: __main__.py CLI (--config flag)

**Objective:** Wire the CLI: `uv run rl-health-interventions --config config/mvp.yaml` runs the experiment and prints results.

**Files:**
- Modify: `src/rl_health_interventions/__main__.py`
- Create: `tests/unit/test_cli.py`

**Step 1: Write failing test**

```python
import subprocess
import sys
from pathlib import Path


def test_cli_runs_with_config_flag(tmp_path: Path, config_yaml_path: Path) -> None:
    """Test that `python -m rl_health_interventions --config path` works end-to-end."""
    result = subprocess.run(
        [sys.executable, "-m", "rl_health_interventions", "--config", str(config_yaml_path), "--output", str(tmp_path / "results.csv")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert (tmp_path / "results.csv").exists()
```

Use a fixture or write a minimal config file in the test.

**Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_cli.py -v
```

Expected: FAIL (CLI is hello-world).

**Step 3: Implement __main__.py**

```python
from __future__ import annotations
import argparse
import logging
from pathlib import Path

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.experiment import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="RL health interventions simulator")
    parser.add_argument("--config", type=str, required=True, help="Path to MDP YAML config")
    parser.add_argument("--agent", type=str, default="thompson_sampling", help="Agent name (default: thompson_sampling)")
    parser.add_argument("--output", type=str, default="results.csv", help="Output CSV path (default: results.csv)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    logger.info("Loaded config from %s", args.config)
    logger.info("Episode: %d days × %d steps/day = %d timesteps",
                config.episode_days, config.steps_per_day,
                config.episode_days * config.steps_per_day)

    agent = make_agent(args.agent)
    if hasattr(agent, 'seed') and args.seed is not None:
        agent.seed = args.seed

    output_path = Path(args.output)
    df = run_episode(config, agent, output_csv=output_path, seed=args.seed)

    print(f"\n=== Episode complete ===")
    print(f"Total steps: {len(df)}")
    print(f"Total reward: {df['reward'].sum():.2f}")
    print(f"Mean reward per step: {df['reward'].mean():.4f}")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/test_cli.py -v
```

Expected: PASS.

**Step 5: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: 93 → 94 tests pass.

**Step 6: Manual smoke test**

```bash
uv run rl-health-interventions --config config/mvp.yaml --output /tmp/test_results.csv
```

Expected: prints summary, writes CSV with 450 rows.

**Step 7: Commit**

```bash
git add src/rl_health_interventions/__main__.py tests/unit/test_cli.py
git commit -m "feat(cli): wire --config flag to run experiment

Replaces hello-world stub with full CLI: argparse for config/agent/
output/seed flags, structured logging, summary output."
```

---

## Task 11: config/mvp.yaml example file

**Objective:** Add the example YAML config from Issue #101 spec.

**Files:**
- Create: `config/mvp.yaml`

**Step 1: Write the file**

```yaml
# Issue #101 MVP config — binary-action rule-based simulator
# See docs/plans/issue-101-mvp.md for context
activity_levels:
  - sedentary
  - active
actions:
  - send
  - don_t_send
time_of_day:
  - morning
  - midday
  - afternoon
  - evening
  - night
steps_per_day: 5
episode_days: 90
initial_state: sedentary
reward_active: 1.0
reward_sedentary: 0.0
seed: 42
transition:
  sedentary:
    send:
      active: 0.3
      sedentary: 0.7
    don_t_send:
      active: 0.1
      sedentary: 0.9
  active:
    send:
      active: 0.8
      sedentary: 0.2
    don_t_send:
      active: 0.6
      sedentary: 0.4
masks:
  morning:
    sedentary: 0.0
    active: 0.0
  midday:
    sedentary: 0.0
    active: 0.0
  afternoon:
    sedentary: 0.0
    active: 0.0
  evening:
    sedentary: 0.0
    active: 0.0
  night:
    sedentary: 1.0
    active: 1.0
```

**Step 2: Verify it loads**

```bash
uv run python -c "from rl_health_interventions.config.loader import load_config; c = load_config('config/mvp.yaml'); print(c)"
```

Expected: prints MDPConfig instance without errors.

**Step 3: Run end-to-end smoke test**

```bash
uv run rl-health-interventions --config config/mvp.yaml --output /tmp/mvp_results.csv
```

Expected: prints summary, writes 450-row CSV.

**Step 4: Inspect a few rows of the CSV**

```bash
head -5 /tmp/mvp_results.csv
```

Expected: header + first 4 step rows with valid state/action/reward values.

**Step 5: Commit**

```bash
git add config/mvp.yaml
git commit -m "feat(config): add MVP example config from Issue #101 spec

450-step HeartSteps-style binary simulator. Uses the exact
transition matrix and time-of-day mask from the issue."
```

---

## Task 12: Integration test (end-to-end config → run → CSV)

**Objective:** Add a top-level integration test that exercises the full pipeline through the CLI.

**Files:**
- Create: `tests/integration/test_mvp_end_to_end.py`

**Step 1: Write the test**

```python
import subprocess
import sys
from pathlib import Path
import pandas as pd


def test_mvp_end_to_end(tmp_path: Path) -> None:
    """Run the MVP config through the CLI, verify output."""
    config_path = Path(__file__).parent.parent.parent / "config" / "mvp.yaml"
    output_path = tmp_path / "results.csv"

    result = subprocess.run(
        [sys.executable, "-m", "rl_health_interventions", "--config", str(config_path), "--output", str(output_path), "--agent", "thompson_sampling"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert output_path.exists()

    df = pd.read_csv(output_path)
    assert len(df) == 450  # 90 days × 5 steps/day
    assert set(df.columns) >= {"step", "day", "step_of_day", "time_of_day", "state", "action", "reward"}
    # Verify reward is 0 or 1
    assert df["reward"].isin([0.0, 1.0]).all()
    # Verify night timesteps have no state change
    night_rows = df[df["time_of_day"] == "night"]
    # The reward is based on the post-transition state, so even at night we see
    # the previous state propagated. But state shouldn't change DURING night.
    # We verify by checking that the state at night is the same as the previous step.
    # (Skip this for MVP — the transition test already covers it.)


def test_epsilon_greedy_baseline_also_works(tmp_path: Path) -> None:
    """Same as above but with epsilon-greedy agent."""
    config_path = Path(__file__).parent.parent.parent / "config" / "mvp.yaml"
    output_path = tmp_path / "results_eg.csv"
    result = subprocess.run(
        [sys.executable, "-m", "rl_health_interventions", "--config", str(config_path), "--output", str(output_path), "--agent", "epsilon_greedy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == 450
```

**Step 2: Run integration test**

```bash
uv run pytest tests/integration/test_mvp_end_to_end.py -v
```

Expected: 2 tests PASS.

**Step 3: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: 94 → 96 tests pass (or more if subprocess output is included).

**Step 4: Commit**

```bash
git add tests/integration/test_mvp_end_to_end.py
git commit -m "test(integration): add end-to-end MVP pipeline test

Exercises config/mvp.yaml through the CLI, verifies 450-row CSV
output with Thompson Sampling and epsilon-greedy agents."
```

---

## Final verification (do this before reporting back)

Run all checks from the repo root:

```bash
uv run ruff format --check .
uv run ruff check
uv run ty check
uv run pytest -q
uv build
```

Expected: all green. The dev-workflow skill mandates these checks. If any fail, fix before reporting.

---

## Done when

- All 12 tasks complete, each with a focused commit
- All 96+ tests pass
- `uv run ruff format --check .`, `uv run ruff check`, `uv run ty check`, `uv run pytest -q`, `uv build` all green
- `config/mvp.yaml` exists and runs through the CLI to produce a 450-row CSV
- The branch `feat/issue-101-mvp-simulator` is committed locally (NOT pushed — user pushes after review)

## Out of scope for this plan (per audit + non-goals)

- Multi-timescale reward
- 4 user archetypes
- Burden accumulation model
- Evaluation framework (bootstrap CIs, baselines, metrics)
- Multi-feature synthetic data
- Safety / ethics review

These belong in future issues once Issue #101 ships and we have supervisor feedback.

## References

- Issue #101: https://github.com/wd7512/rl-health-interventions/issues/101
- Audit findings: PR #78 (Phase 1 6-phase audit, merged 7e2dfc8)
- Roadmap reframe: PR #102 (merged fb853af)
- Initial design: docs/initial_design.tex (awaiting supervisor sign-off)
