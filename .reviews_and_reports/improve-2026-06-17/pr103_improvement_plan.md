# PR #103 — Improved Architectural Plan (v1)
## feat: implement MVP binary-action simulator

**Status:** Pre-merge design resolution (v1, revised via grill-me session)
**Last updated:** 17 June 2026

---

## Scope

**Single PR** — no follow-up split. Everything listed here ships together.
**Target config format:** `rule_based.yaml` (states/actions/transition_model). `mvp.yaml` is deleted.
**Branch:** `feat/issue-101-mvp-simulator` (PR #103 already open against `main`)

---

## 1. Decisions Log

| Item | Decision |
|---|---|
| `time_of_day` as state | **Removed.** `reward_multiplier_by_step` replaces it entirely |
| Masking mechanism | **`reward_multiplier_by_step` is the mask.** Optional field; absent means uniform `1.0` multiplier. A `0` signals "no reward here" — no separate mask class or block needed |
| Config-first | **Yes** — string-keyed states/actions, enums removed, cross-reference validators added in same pass |
| Swappable transition | **Yes** — `transition_model.type` already present; registry factory exists already |
| `agents` in YAML | **Yes** — including hyperparams; YAML is the full experiment record |
| `version` field | **No** — see rationale in v0 §2.1 (unchanged) |
| `schemas.py` rename | **Not needed.** File stays `schemas.py`, contents rewritten (enums removed, new config model) |
| `experiment.py` rename | **Not needed.** Stays `experiment.py` |
| `mvp.yaml` | **Delete.** Deprecated new-format file |
| `rule_based.yaml` | **Update.** Add `agents` block |
| `rule_based_with_mask.yaml` | **Update.** Add `reward_multiplier_by_step: [1,1,1,1,0]` + `agents` block |
| `learned_aspirational.yaml`, `llm_based_aspirational.yaml` | **Keep.** Schema-reference mode stubs handle them gracefully |
| `reward_multiplier_by_step` | **Optional field.** Only active when present. Defaults to uniform `1.0` when absent. Precomputed in config. |
| Cross-reference validators | **In scope.** Structural checks on states, actions, transition probs, agents |
| Schema-reference mode stub | **In scope.** `NotImplementedError` seam |
| Transition registry | **Already exists.** `environment.py` must use it instead of direct imports |
| Seed independence (H3) | **Deferred.** Not in this PR |
| Enum removal | **In scope.** Remove `ActivityLevel`, `TimeOfDay`, `Action` enums from all files |
| Regression test | Total reward per agent per seed, 6dp, JSON fixture at seed 42 |
| PDF | Rebuild with UCB results before merge |
| README | Revert to near-original; only the delta this PR directly requires |

---

## 2. Design Rationale

### 2.1 No `version` field

Version fields in config files are appealing but rarely earn their place at this stage:

- **Git is the version history.** Any breaking schema change is traceable to a commit.
- **The validator catches mismatches at load time.** A config missing a required field, or containing an unknown key, will fail loudly without needing a version string to identify it.
- **YAGNI.** If the schema ever diverges enough to need migration tooling, that's the moment to add a version field — and at that point you can default unversioned files to `"1.0"` for backward compatibility.

Skip it for now.

### 2.2 `reward_multiplier_by_step` as the mask

`reward_multiplier_by_step` is a soft mask: the agent *can* take any action at any step, but gets zero reward at masked steps. This is preferable to a hard action mask at this stage because:

- Simpler — no separate mask class, registry, or YAML block
- Pedagogically honest — the constraint is expressed as a reward signal, which is how RL naturally handles time-varying incentives
- Agents learn not to nudge at zero-reward steps; a random baseline will still nudge there, which is the correct baseline behaviour

The distinction between `rule_based.yaml` (no mask) and `rule_based_with_mask.yaml` (step 4 masked) is entirely captured by the value of this array. Everything else is identical.

### 2.3 `agents` as experiment record

Placing agents (with hyperparams) in the YAML means a single file fully specifies an experiment: same config → same results, regardless of who runs it or when. This is especially important for the regression test and for reproducibility in the eventual publication context.

### 2.4 Enum removal rationale

Removing `ActivityLevel`, `TimeOfDay`, and `Action` enums makes the config model purely string-keyed. Benefits:

- Config YAML can use any strings (e.g., `sedentary`/`active`, `nudge`/`idle`), not just enum members
- Adding new states/actions requires only a YAML change, not code changes
- Schema-reference mode (`states: {schema: heartsteps}`) works naturally with string resolution
- Simplifies the `StateView` to carry plain strings instead of enum members

This touches every file that references the enums (~20 files) but each change is mechanical: `ActivityLevel.SEDENTARY` → `"sedentary"`, `Action.SEND` → `"nudge"` (as defined by the config), etc.

---

## 3. Target Config Files

### 3.1 `config/rule_based.yaml` — Baseline, no masking

Clean environment baseline. No `reward_multiplier_by_step` — absent means uniform 1.0.

```yaml
episode_days: 90
steps_per_day: 5
seed: 42
initial_state: sedentary

states:
  sedentary:
    reward: 0.0
  active:
    reward: 1.0

actions:
  - nudge
  - idle

transition_model:
  type: rule_based
  transition_probabilities:
    sedentary:
      nudge:
        active: 0.3
        sedentary: 0.7
      idle:
        active: 0.1
        sedentary: 0.9
    active:
      nudge:
        active: 0.5
        sedentary: 0.5
      idle:
        active: 0.6
        sedentary: 0.4

agents:
  - type: thompson_sampling
    alpha_prior: 1
    beta_prior: 1
```

### 3.2 `config/rule_based_with_mask.yaml` — Step 4 masked

Identical to `rule_based.yaml` except with `reward_multiplier_by_step`. Step 4 represents end-of-day; nudging there yields no reward. Agents that learn will stop nudging at step 4.

```yaml
# ... same as rule_based.yaml up to transition_model ...

# Step 4 is end-of-day — zero multiplier acts as a soft mask.
reward_multiplier_by_step: [1, 1, 1, 1, 0]

agents:
  - type: thompson_sampling
    alpha_prior: 1
    beta_prior: 1
```

### 3.3 Forward reference: `learned_aspirational.yaml` / `heartsteps_learned.yaml`

Out of scope for this PR, but confirmed the YAML shape is consistent with the learned-model path. The schema-reference mode stub handles it gracefully with `NotImplementedError`.

```yaml
episode_days: 90
steps_per_day: 5
seed: 42
initial_state: default

states:
  schema: heartsteps
actions:
  schema: heartsteps

transition_model:
  type: learned
  model: monte-carlo
  schema: heartsteps

reward_multiplier_by_step: [1, 1, 1, 1, 0]

agents:
  - type: thompson_sampling
```

---

## 4. Validator Specification

Validation runs at config load time. A bad config must fail with a specific, actionable message.

### 4.1 Structural checks (all configs)

```
episode_days             — positive integer
steps_per_day            — positive integer
seed                     — integer
initial_state            — in states (inline) OR == "default" (schema-ref mode)
```

### 4.2 Inline-mode cross-reference checks

Run these when `states` is a dict (not `{schema: ...}`).

```
states
  - non-empty dict
  - each entry has a numeric `reward` field

actions
  - non-empty list of strings
  - no duplicates

transition_model.transition_probabilities
  - outer keys (states) ⊆ declared states
  - each state must have an entry for every declared action
  - each action's next-state keys ⊆ declared states
  - each next-state dict must cover all declared states
  - each probability row sums to 1.0 (tolerance: 1e-6)

reward_multiplier_by_step (optional)
  - if present: is a list, len == steps_per_day, all values are numeric
  - if absent: defaults to uniform [1.0, ..., 1.0]

agents
  - each entry has a `type` field
  - known types: {thompson_sampling, random, epsilon_greedy, ucb}
  - thompson_sampling: requires alpha_prior > 0, beta_prior > 0
  - epsilon_greedy:    requires epsilon ∈ (0, 1]
  - ucb:               requires c > 0
```

### 4.3 Schema-reference mode

When `states` contains a `schema` key, skip inline cross-reference checks and log a warning that validation is deferred to the schema registry. Stub the registry loader with `NotImplementedError("Schema registry not yet implemented")`. This seam exists in this PR so the learned-model path doesn't need to touch the validator later.

---

## 5. `reward_multiplier_by_step` Integration

`reward_multiplier_by_step` is **optional**. When absent, the config model defaults to a uniform multiplier of `1.0` for all steps (no effect).

When present, the config model precomputes a `per_step_reward` array via a Pydantic `@model_validator(mode='after')`:

```python
@model_validator(mode='after')
def _compute_per_step_reward(self) -> Self:
    multiplier = self.reward_multiplier_by_step or [1.0] * self.steps_per_day
    self.per_step_reward = [
        {state: data.reward * mult for state, data in self.states.items()}
        for mult in multiplier
    ]
    return self
```

The reward handler receives `(state_name, step_idx)` and returns the precomputed value. No runtime multiplication or branching per step.

The transition system is unaffected — transitions always run using the probability matrix. Only the reward value is scaled.

---

## 6. Transition Registry Usage

`environment.py` currently imports `RuleBasedTransition` and `CompoundReward` directly. Change to use the existing registry factories:

```python
from rl_health_interventions import make_transition, make_reward

self.transition = make_transition(config)
self.reward = make_reward(config)
```

Both `make()` factories are already registered in the package `__init__.py`. The factories must be updated to read from the new config shape.

---

## 7. Full File Manifest

### Delete
| File | Reason |
|---|---|
| `config/mvp.yaml` | Deprecated new-format config |

### Update (config)
| File | Changes |
|---|---|
| `config/rule_based.yaml` | Add `agents` block |
| `config/rule_based_with_mask.yaml` | Add `reward_multiplier_by_step: [1,1,1,1,0]` + `agents` block |

### Keep (unchanged)
| File | Reason |
|---|---|
| `config/learned_aspirational.yaml` | Schema-ref stub handles it at load time |
| `config/llm_based_aspirational.yaml` | Schema-ref stub handles it at load time |

### Rewrite (source)
| File | Changes |
|---|---|
| `src/rl_health_interventions/config/schemas.py` | New config model: string-keyed, enums removed, cross-reference validators, schema-ref stub, precomputed `per_step_reward` |
| `src/rl_health_interventions/config/loader.py` | Update to new config model |
| `src/rl_health_interventions/state.py` | Remove `TimeOfDay`; `activity: str` instead of enum |
| `src/rl_health_interventions/environment.py` | No time-of-day; use `make_transition()` + `make_reward()`; pass step index |
| `src/rl_health_interventions/transitions/rule_based.py` | Remove mask checking; string keys; accept config |
| `src/rl_health_interventions/transitions/_base.py` | `ActivityLevel` → `str`, `Action` → `str`, remove `TimeOfDay` |
| `src/rl_health_interventions/rewards/compound.py` | Use precomputed `per_step_reward`; accept step index |
| `src/rl_health_interventions/rewards/_base.py` | Update ABC signature |
| `src/rl_health_interventions/agents/_base.py` | `Action` → `str` |
| `src/rl_health_interventions/agents/*.py` (4 files) | `Action` → `str` |
| `src/rl_health_interventions/experiment.py` | New config shape; no time_of_day in state |
| `src/rl_health_interventions/__main__.py` | Config path defaults to `rule_based.yaml` |
| `tests/conftest.py` | Update fixtures |
| `tests/unit/config/test_mdp_config.py` | New schema tests |
| `tests/unit/test_state.py` | Remove time_of_day tests |
| `tests/unit/test_environment.py` | No time-of-day tests; add multiplier tests |
| `tests/unit/test_experiment.py` | New config state |
| `tests/unit/transitions/test_rule_based_transition.py` | No mask tests |
| `tests/unit/rewards/test_compound_reward.py` | Precomputed per-step reward |
| `tests/unit/agents/test_*.py` | Enum → string |
| `tests/integration/test_mvp_end_to_end.py` | Config path → `rule_based.yaml` |

### Create
| File | Purpose |
|---|---|
| `tests/unit/config/test_mdp_validators.py` | Cross-reference validator tests |
| `tests/fixtures/mvp_expected_rewards.json` | Regression fixture (generated at seed 42) |
| `scripts/generate_regression_fixture.py` | One-shot generator for the fixture |

---

## 8. Regression Test

```python
# tests/test_regression_mvp.py

import json, pytest
from pathlib import Path
from rl_health_interventions.experiment import run_experiment

FIXTURE = Path("tests/fixtures/mvp_expected_rewards.json")

def test_mvp_total_rewards_unchanged():
    results = run_experiment("config/rule_based.yaml")
    expected = json.loads(FIXTURE.read_text())
    for agent_type, expected_reward in expected.items():
        assert results[agent_type] == pytest.approx(expected_reward, abs=1e-6), (
            f"Agent '{agent_type}' reward changed: "
            f"got {results[agent_type]}, expected {expected_reward}"
        )
```

**Generate the fixture once, commit it:**

```python
# scripts/generate_regression_fixture.py
import json
from pathlib import Path
from rl_health_interventions.experiment import run_experiment

results = run_experiment("config/rule_based.yaml")
Path("tests/fixtures/mvp_expected_rewards.json").write_text(
    json.dumps(results, indent=2)
)
```

Fixture shape (values populated by running the generator):
```json
{
  "thompson_sampling": 0.0
}
```

---

## 9. Merge Checklist

- [ ] `config/mvp.yaml` deleted
- [ ] `config/rule_based.yaml` updated with `agents` block
- [ ] `config/rule_based_with_mask.yaml` updated with `reward_multiplier_by_step` + `agents` block
- [ ] `schemas.py` rewritten (enums removed, new config model, validators, stub)
- [ ] `state.py` — `TimeOfDay` removed, `activity` is `str`
- [ ] `environment.py` — no time-of-day, uses registry factories, passes step index
- [ ] `transitions/rule_based.py` — no mask, string keys
- [ ] `rewards/compound.py` — precomputed per-step array
- [ ] All agents — `Action` → `str`
- [ ] All tests pass: `uv run ruff check`, `uv run ty check`, `uv run pytest`
- [ ] Regression test written; fixture committed
- [ ] `learned_aspirational.yaml`, `llm_based_aspirational.yaml` load without error (schema-ref stub)
- [ ] Smoke test: `uv run rl-health-interventions --config config/rule_based.yaml`
- [ ] PDF rebuilt to include UCB results
- [ ] README reverted to near-original; only delta this PR directly requires

---

## 10. Open Questions

Two questions deferred, both related to the future learned-model path. Neither is a blocker.

**OQ-1 — `initial_state: default` semantics**
In schema-reference mode, `default` presumably means "sample from the empirical initial-state distribution". Resolve when learned transition type is implemented.

**OQ-2 — Schema registry resolution**
When `states: {schema: heartsteps}` is encountered, where does the resolver look? Options: (a) `config/schemas/heartsteps.yaml` file, (b) Python module, (c) registry dict. The stub `NotImplementedError` just needs to name the preferred direction.
