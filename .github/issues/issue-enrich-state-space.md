# Issue 2: Enrich state space with user context

**Labels:** `enhancement`, `phase-1`
**Assignees:** @wd7512

## Context

Design doc review with Mengyan Zhang (16 Jun 2026). Current `StateView` contains only `activity, day, step_of_day, steps_per_day` — only `activity` is a real state variable. Mengyan asked: *"for the state space, can we add some user-related components in?"*

The `initial_design.tex §3` defines 12+ potential state variables across four categories.

## Current state

```python
@dataclass(frozen=True)
class StateView:
    activity: str        # "sedentary" | "active"
    day: int
    step_of_day: int
    steps_per_day: int
```

The contextual bandit base (`_get_context_key`) only supports `state.activity` as a context feature.

## Specification

### Target state representation (from `initial_design.tex`)

```
s_t = { x_t_wearable, x_t_context, x_t_history, x_profile }
```

| Category | Variables | Source |
|----------|-----------|--------|
| Wearable | `steps`, `heart_rate`, `sleep_hours` | `SyntheticDataGenerator` or `ResponseModel` |
| Context | `time_of_day`, `day_of_week` | Derived from step index |
| History | `previous_action`, `previous_response`, `goal_progress`, `burden` | Environment internals |
| Profile | `archetype`, `baseline_activity`, `age`, `gender` | `UserProfile` (sampled at episode start) |

### Phase 1 scope (this issue) — minimum viable enrichment

For a first iteration, add:

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `archetype` | `str` | Config-driven (uniform random at reset) | One of: goal_driven, social, resistant, stable |
| `burden` | `float` | Environment accumulator | Starts at 0, incremented by action penalties |
| `goal_progress` | `float` | Computed from step history | Ratio of current to target steps |
| `steps` | `int` | Simulated via `ResponseModel` | Placeholder: drawn from synthetic distribution |
| `previous_action` | `str` | Environment internals | Tracked by `Environment.step()` |

### Required changes

1. **`state.py`** — Extend `StateView`:
   ```python
   @dataclass(frozen=True)
   class StateView:
       activity: str
       day: int
       step_of_day: int
       steps_per_day: int
       global_step: int = field(init=False)
       archetype: str = "unknown"
       burden: float = 0.0
       goal_progress: float = 0.0
       steps: int = 0
       previous_action: str | None = None
   ```

2. **`environment.py`** — Update `step()` to construct enriched state:
   - Track `previous_action` internally
   - Compute `goal_progress` from accumulated steps vs target (configurable baseline)
   - Call `ResponseModel.response()` to get simulated step counts
   - Sample `archetype` uniformly at `reset()` from the 4 types

3. **Config schema** — Add optional user profile fields to `MDPConfig`:
   ```yaml
   user_profile:
     baseline_steps: 8000
     target_steps: 10000
     archetype_sampling: uniform  # or config-driven
   ```

4. **Contextual bandits** — `_get_context_key` already uses `hasattr(state, self.context_feature)`. New fields are immediately usable by setting `context_feature: archetype` or `context_feature: burden` in agent config.

## Deliverables

1. Updated `StateView` with profile, burden, goal_progress fields
2. Updated `Environment.step()` to construct enriched state
3. Config-driven archetype sampling at episode start
4. Updated `config/rule_based.yaml` with default user_profile section
5. Test: verify new StateView fields are populated correctly and agents can condition on them
6. Run `uv run ruff check && uv run pytest` — all pass

## Out of scope
- Full `UserProfile` Pydantic schema with 4 archetype parameter sets (M-04 / Issue #3)
- Burden accumulation/decay model (M-04)
- Multi-feature synthetic data for HR/sleep (M-07)

## Related
- `docs/design/initial_design.tex §3` (full state definition)
- Issue #1 (action extension — complements enriched state)
- Issue #3 (4 persona matrices — uses archetype field)
- `src/rl_health_interventions/state.py`
- `src/rl_health_interventions/environment.py`
