# Epic: State-Action-Reward Extension

**Epic branch:** `epic/state-action-reward`

**Parent:** `main`

**Status:** `[ ]` In progress

---

## Vision

Extend the MVP (2 states, 2 actions, simple reward) into a configurable
simulation with 6 design-doc actions, 4 continuous state dimensions, and
a multi-timescale reward with delayed signals. Deliver 7 experiment
configurations covering every combination of the three extensions.

---

## Structure

```
main
  |
  └── epic/state-action-reward
        |        |        |           |        |        |        |
      pr1a      pr1b     pr1c      pr4(1a1b) pr5(1a1c) pr6(1b1c) pr7(full)
```

---

## Config Matrix

| # | Config file | Actions | States | Reward |
|---|-------------|---------|--------|--------|
| - | `config/rule_based.yaml` (MVP) | 2 (nudge, idle) | 2 (sedentary, active) | Simple per-state |
| 1 | `config/experiments/1a_actions_only.yaml` | 6 (design doc) | 2 | Simple per-state |
| 2 | `config/experiments/1b_states_only.yaml` | 2 | 4 continuous | Simple per-state |
| 3 | `config/experiments/1c_reward_only.yaml` | 2 | 2 | Multi-timescale |
| 4 | `config/experiments/1a_1b.yaml` | 6 | 4 continuous | Simple per-state |
| 5 | `config/experiments/1a_1c.yaml` | 6 | 2 | Multi-timescale |
| 6 | `config/experiments/1b_1c.yaml` | 2 | 4 continuous | Multi-timescale |
| 7 | `config/experiments/full.yaml` | 6 | 4 continuous | Multi-timescale |

---

## Phase 1 - Individual Extensions (3 PRs, parallel from main)

Each branches from `main`, extends one dimension from the MVP baseline, and
PRs into `epic`. All must maintain MVP backward compatibility:
`config/rule_based.yaml` must produce identical results after each PR
(except regression fixture regeneration when stochastic paths change).

---

### PR 1: `feat/1a-actions` - 6-Action Space

**Objective:** Extend `actions` from 2 MVP strings to 6 design-doc actions
with metadata (reward_penalty, burden_penalty).

**Config schema (`config/schemas.py`):**

`actions` field accepts either `list[str]` (MVP mode, e.g.
`["nudge", "idle"]`) or `list[dict]` (extended mode):
```yaml
actions:
  - name: no_message
    reward_penalty: 0
    burden_penalty: 0
  - name: motivational_prompt
    reward_penalty: 0.1
    burden_penalty: 0.2
  - name: walking_suggestion
    reward_penalty: 0.2
    burden_penalty: 0.3
  - name: goal_reminder
    reward_penalty: 0.15
    burden_penalty: 0.2
  - name: recovery_suggestion
    reward_penalty: 0.1
    burden_penalty: 0.25
  - name: progress_feedback
    reward_penalty: 0.05
    burden_penalty: 0.15
```
- `_cross_reference_validators`: extract `name` from dict entries for
  transition probability checks. Reject duplicate names.
- `_compute_per_step_reward`: no change (reward handler currently ignores
  actions - handled in PR 1c).

**RuleBasedTransition:** no code change needed (action strings stay unique
identifiers; transition matrices index by action name).

**Agents:** no code change (all agents use `self._actions: list[str]` -
extract action names from dicts at construction time).

**Config files:**
- `config/experiments/1a_actions_only.yaml` - 6 actions, 2 MVP states,
  6x2=12 transition probability entries

**Tests:**
- `test_actions_as_dict_list_accepted` - list of dicts parses correctly
- `test_actions_dict_list_duplicate_rejected` - duplicate action names rejected
- `test_actions_dict_list_missing_name_rejected` - dict without `name` rejected
- `test_actions_mvp_string_list_still_works` - `["nudge", "idle"]` still parses
- `test_actions_6_full_transition_coverage` - all 6x2 transition entries present
- `test_environment_step_with_6_actions` - env step returns valid output

**Regression:** regenerate `mvp_expected_rewards.json` (RNG seed interactions
with action list size change reward path).

---

### PR 2: `feat/1b-states` - 4D Continuous State

**Objective:** Replace 2 discrete MVP states (sedentary/active) with 4
continuous dimensions: steps, weight, time_of_day, day_of_week.

**State variables:**

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `activity` | `str` | - | MVP compat: "sedentary" \| "active" |
| `steps` | `float` | >= 0 | Step count since last decision epoch |
| `weight` | `float` | > 0 | Current body weight (kg) |
| `time_of_day` | `int` | 0-23 | Current hour (deterministic from step counter) |
| `day_of_week` | `int` | 0-6 | 0=Sunday (deterministic from day counter) |
| `day` | `int` | >= 0 | Episode day (existing MVP field) |
| `step_of_day` | `int` | 0..steps_per_day-1 | Step within day (existing) |
| `steps_per_day` | `int` | >= 1 | Config parameter (existing) |

**StateView changes (`state.py`):**
```python
@dataclass(frozen=True)
class StateView:
    activity: str
    day: int
    step_of_day: int
    steps_per_day: int = 5
    steps: float | None = None
    weight: float | None = None
    time_of_day: int | None = None
    day_of_week: int | None = None

    @property
    def global_step(self) -> int:
        return self.day * self.steps_per_day + self.step_of_day
```
New fields default to `None` so MVP tests constructing StateView with only
the original 4 fields continue to work without modification.

**Transition ABC change (`transitions/_base.py`):**
```python
class TransitionModel(ABC):
    @abstractmethod
    def transition(self, state: StateView, action: str) -> StateView: ...
```
Changed from `(state: str, action: str) -> str`.

**RuleBasedTransition update (`transitions/rule_based.py`):**
Two modes detected from config: **MVP mode** (no `state_dynamics`) reads
`state.activity`, samples transition matrix, returns
`dataclasses.replace(state, activity=next_activity)`. **Extended mode**
(`state_dynamics` present) evolves all fields.

**Reward ABC change (`rewards/_base.py`):**
```python
class RewardHandler(ABC):
    @abstractmethod
    def reward(self, state: StateView, action: str, step_idx: int) -> tuple[float, bool]: ...
```
Changed from `state: str` to `state: StateView`. CompoundReward in MVP
mode reads `state.activity` - identical behavior.

**Transition dynamics config schema:**
```yaml
state_dynamics:
  steps:
    response_multiplier:
      no_message: 0
      motivational_prompt: 200
      walking_suggestion: 500
      goal_reminder: 300
      recovery_suggestion: -100
      progress_feedback: 400
    tod_modulation:
      0: 0.2
      6: 0.5
      12: 0.9
      18: 0.6
      23: 0.2
    dow_modulation:
      0: 0.7
      1: 0.9
      5: 0.8
      6: 0.6
    noise_std: 50
  weight:
    meal_effect:
      0: -0.02
      6: 0.01
      12: 0.03
      18: 0.03
      23: -0.01
    weekend_boost: 0.05
    steps_coefficient: -0.0001
    noise_std: 0.05
```

**Equations:**
```
tod = step_of_day * (24 / steps_per_day)
dow = day % 7
steps_mean = response_multiplier[action] * tod_modulation[tod] * dow_modulation[dow]
steps_{t+1} = steps_t + steps_mean + N(0, noise_std)
weight_{t+1} = weight_t + meal_effect[tod]
  + (weekend_boost if dow in {0,6} else 0)
  + steps_coefficient * (steps_{t+1} - steps_t)
  + N(0, weight.noise_std)
```

**Environment changes (`environment.py`):**
- `reset()`: initialize steps, weight from configurable baselines.
  time_of_day/day_of_week computed deterministically.
- `step()`: pass full `StateView` to transition. Use returned `StateView`
  as next state. Overwrite tod/dow from step counter since deterministic.

**Config files:**
- `config/experiments/1b_states_only.yaml` - 2 MVP actions, 4 continuous
  states, state_dynamics block

**Tests:**
- `test_state_view_full_construction` - all 8 fields set, frozen
- `test_state_view_mvp_backward_compat` - 4-field construction still works
- `test_environment_reset_continuous` - reset creates StateView with steps/weight
- `test_steps_evolves_per_equation` - steps changes by expected amount
- `test_weight_evolves_per_equation` - weight changes with tod/dow/steps
- `test_time_of_day_deterministic` - formula matches expectation
- `test_day_of_week_deterministic` - dow = day % 7
- `test_mvp_config_produces_identical_reward` - backward compat verified

---

### PR 3: `feat/1c-reward` - Multi-Timescale Reward

**Objective:** Add delayed reward every N steps while keeping MVP simple
per-state reward as default.

**Config schema (`config/schemas.py`):**
New optional block:
```yaml
reward_weights:
  mode: multi_timescale
  delayed_reward_interval: 21
  delayed_reward_value: 10.0
  delayed_reward_scale: 5.0       # optional: for scaling bonus
  delayed_reward_threshold: 0.6   # optional: min active rate
```
When `reward_weights` is absent -> mode defaults to `"simple"`.

**CompoundReward update (`rewards/compound.py`):**
- Accept `StateView` instead of `str` (from ABC change in PR 1b).
- **Simple mode:** `reward = per_step_reward[step_idx][state.activity]`
- **Multi-timescale (Option A - flat):**
  ```
  bonus = delayed_reward_value if (global_step > 0 and
          global_step % interval == 0) else 0
  ```
- **Multi-timescale (Option B - scaled):**
  ```
  if global_step % interval == 0:
      active_rate = recent_active_count / interval
      bonus = delayed_reward_scale * active_rate if active_rate >= threshold else 0
  ```
  Track recent activity via cumulative count reset each interval.

**Environment:** pass `state.global_step` to reward handler.

**Config files:**
- `config/experiments/1c_reward_only.yaml` - 2 MVP actions, 2 MVP states,
  reward_weights with both options

**Tests:**
- `test_simple_mode_mvp_identical` - no reward_weights -> same as current
- `test_flat_bonus_at_interval` - Option A bonus at boundary
- `test_no_bonus_off_interval` - Option A at non-boundary
- `test_scaled_bonus_depends_on_activity` - Option B scale x active rate
- `test_scaled_bonus_zero_below_threshold` - Option B below threshold

---

## Phase 2 - Pairwise Integrations (3 PRs, from epic after Phase 1)

Each branches from `epic` after Phase 1 merges, adds combined config and
integration tests, then PRs back to `epic`.

### PR 4: `int/1a-1b` - 6 Actions + 4 Continuous States

- `config/experiments/1a_1b.yaml` - 6 actions, 4 continuous states,
  state_dynamics with entries for all 6 action keys
- Integration test: end-to-end run, verify steps/weight evolve with 6 actions
- Code changes: none expected (transition equations use action names as keys)

### PR 5: `int/1a-1c` - 6 Actions + Multi-Timescale Reward

- `config/experiments/1a_1c.yaml` - 6 actions with reward_penalties,
  2 MVP states, reward_weights
- Integration test: reward_penalty subtracted from base reward at intervals
- Code change (~5 lines in CompoundReward):
  `penalty = self._action_penalties.get(action, 0.0); reward = base + bonus - penalty`
  When action metadata absent (MVP mode) -> penalty = 0

### PR 6: `int/1b-1c` - 4 Continuous States + Multi-Timescale Reward

- `config/experiments/1b_1c.yaml` - 2 MVP actions, 4 continuous states,
  reward_weights with Option B
- Integration test: delayed reward scales with state.steps trajectory
- Code change (~3 lines): Option B reads `state.steps` for activity rate

---

## Phase 3 - Full Integration

### PR 7: `int/full` - All Three Together

- `config/experiments/full.yaml` - 6 actions, 4 continuous states, reward_weights
- `scripts/run_cross_experiment.py` - run all 7 configs, print comparison table
- Integration test `tests/integration/test_cross_experiment.py` - all 7 configs
  load and run without errors
- Code changes: none expected

---

## Verification Per PR

```bash
uv run ruff format --check .
uv run ruff check
uv run ty check --exclude tests/
uv run pytest
```

## Risks

| Risk | Mitigation |
|------|-----------|
| ABC signature change in PR 1b touches ~15 test files | Mechanical wrapping. Each test file change is 1-2 lines. |
| Regression fixture churn from RNG seed interaction | Freeze MVP config test path; assert MVP config produces same reward independently of extended configs. |
| Action name alignment between PRs | Define action names as constants in PR 1a; 1b/1c configs reference them by name. |
| time_of_day calculation | Use `int(step_of_day * 24 / steps_per_day)`. Document as approximation. |
