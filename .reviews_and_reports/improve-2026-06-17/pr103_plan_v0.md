# PR #103 — Architectural Plan
## feat: implement MVP binary-action simulator

**Status:** Pre-merge design resolution  
**Last updated:** 17 June 2026

---

## 1. Decisions Log

| Item | Decision |
|---|---|
| `time_of_day` as state | **Removed.** `reward_multiplier_by_step` replaces it entirely |
| Masking mechanism | **`reward_multiplier_by_step` is the mask.** A multiplier of `0` at a step signals "no reward here" — no separate mask class or block needed |
| Config-first | **Yes** — string-keyed states/actions, enums removed, cross-reference validators added in same pass |
| Swappable transition | **Yes** — `transition_model.type` already present; build registry factory now with one entry |
| `agents` in YAML | **Yes** — including hyperparams; YAML is the full experiment record |
| `version` field | **No** — see rationale in §2.1 |
| `experiment.py` | **Rename** to `mvp_runner.py` (hardcoded to `mvp.yaml`); a generic `runner.py` is a separate future file |
| `schemas.py` | **Rename** to `rule_based_schema.py` now; revert to `schemas.py` only after enums are fully removed |
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

`reward_multiplier_by_step` is a soft mask: the agent *can* take any action at any step, but
gets zero reward at masked steps. This is preferable to a hard action mask at this stage because:

- Simpler — no separate mask class, registry, or YAML block
- Pedagogically honest — the constraint is expressed as a reward signal, which is how RL naturally handles time-varying incentives
- Agents learn not to nudge at zero-reward steps; a random baseline will still nudge there, which is the correct baseline behaviour

The distinction between `mvp.yaml` (no mask) and `mvp_with_mask.yaml` (step 4 masked) is
entirely captured by the value of this array. Everything else is identical.

> **Note:** The existing branch has `reward_multiplier_by_step: [1, 1, 1, 1, 0]` in `mvp.yaml`.
> The plan below separates this into two files — `mvp.yaml` with a uniform multiplier and
> `mvp_with_mask.yaml` with the step-4 zero. This requires a small change to the existing file.

### 2.3 `agents` as experiment record

Placing agents (with hyperparams) in the YAML means a single file fully specifies an experiment:
same config → same results, regardless of who runs it or when. This is especially important for
the regression test and for reproducibility in the eventual publication context.

---

## 3. Draft Config Files

### 3.1 `mvp.yaml` — Baseline, no masking

All steps are equally rewarded. Clean environment baseline.

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

# All steps are equally rewarded in the base MVP.
reward_multiplier_by_step: [1, 1, 1, 1, 1]

agents:
  - type: random
  - type: epsilon_greedy
    epsilon: 0.1
  - type: ucb
    c: 2.0
  - type: thompson_sampling
```

### 3.2 `mvp_with_mask.yaml` — Step 4 masked

Identical to `mvp.yaml` except `reward_multiplier_by_step`. Step 4 represents end-of-day;
nudging there yields no reward. Agents that learn will stop nudging at step 4.

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

# Step 4 is end-of-day — zero multiplier acts as a soft mask.
# Length must equal steps_per_day.
reward_multiplier_by_step: [1, 1, 1, 1, 0]

agents:
  - type: random
  - type: epsilon_greedy
    epsilon: 0.1
  - type: ucb
    c: 2.0
  - type: thompson_sampling
```

### 3.3 Forward reference: `heartsteps_learned.yaml`

Out of scope for this PR, but included to confirm the YAML shape is consistent
with the learned-model path. No changes needed to the runner or validator to
accommodate this config once the `learned` transition type is implemented.

```yaml
episode_days: 90
steps_per_day: 5
seed: 42
initial_state: default   # reserved keyword: sample from learned initial-state distribution

# Schema-reference mode: states and actions resolved from a named schema registry.
# Validator defers cross-reference checks to the schema loader.
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

Validation runs at config load time. A bad config must fail with a specific,
actionable message — not silently corrupt an experiment at step 47.

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
  - each next-state dict must cover all declared states (no missing targets)
  - each probability row sums to 1.0 (tolerance: 1e-6)

reward_multiplier_by_step
  - is a list
  - len == steps_per_day
  - all values are numeric

agents
  - each entry has a `type` field
  - known types: {random, epsilon_greedy, ucb, thompson_sampling}
  - epsilon_greedy: requires epsilon ∈ (0, 1]
  - ucb:            requires c > 0
```

### 4.3 Schema-reference mode

When `states` contains a `schema` key, skip inline cross-reference checks and log a warning
that validation is deferred to the schema registry. Stub the registry loader with
`NotImplementedError("Schema registry not yet implemented")`. This seam exists in this PR
so the learned-model path doesn't need to touch the validator later.

---

## 5. Transition Registry (Strategy Pattern)

New file: `transitions/__init__.py`. `environment.py` calls `build_transition(config)`
and never imports a concrete transition class directly.

```python
# transitions/__init__.py

from rl_health_interventions.transitions.rule_based import RuleBasedTransition

TRANSITION_REGISTRY = {
    "rule_based": RuleBasedTransition,
    # "learned": LearnedTransition,   — future
    # "llm":     LLMTransition,       — future
}

def build_transition(config):
    transition_type = config.transition_model.type
    if transition_type not in TRANSITION_REGISTRY:
        raise ValueError(
            f"Unknown transition type: '{transition_type}'. "
            f"Available: {list(TRANSITION_REGISTRY.keys())}"
        )
    return TRANSITION_REGISTRY[transition_type](config)
```

Adding an LLM transition later is a one-line registry entry and a new YAML `type: llm` string.

---

## 6. File Rename and Restructure Plan

Rename first, in a dedicated commit, before any logic changes.

```
# Step 1: Renames (git mv — separate commit)
src/.../schemas.py      → src/.../rule_based_schema.py
src/.../experiment.py   → src/.../mvp_runner.py

# Step 2: New files
src/.../transitions/__init__.py    — transition registry + build_transition()
config/mvp_with_mask.yaml

# Step 3: Modify existing files
config/mvp.yaml
  - reward_multiplier_by_step: [1,1,1,1,1]  (was [1,1,1,1,0] — moved to mvp_with_mask)
  - add agents block

src/.../environment.py
  - replace hardcoded transition import with build_transition(config)

src/.../transitions/rule_based.py
  - remove 4-item import; accept config object directly

src/.../rule_based_schema.py  (formerly schemas.py)
  - add cross-reference validators from §4.2
  - add schema-reference mode stub from §4.3
```

---

## 7. Regression Test

```python
# tests/test_regression_mvp.py

import json, pytest
from pathlib import Path
from rl_health_interventions.mvp_runner import run_experiment

FIXTURE = Path("tests/fixtures/mvp_expected_rewards.json")

def test_mvp_total_rewards_unchanged():
    results = run_experiment("config/mvp.yaml")
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
from rl_health_interventions.mvp_runner import run_experiment

results = run_experiment("config/mvp.yaml")
Path("tests/fixtures/mvp_expected_rewards.json").write_text(
    json.dumps(results, indent=2)
)
```

Fixture shape (values populated by running the generator):
```json
{
  "random": 0.0,
  "epsilon_greedy": 0.0,
  "ucb": 0.0,
  "thompson_sampling": 0.0
}
```

---

## 8. Merge Checklist

### Blockers (must be in this PR)
- [ ] PDF rebuilt to include UCB results
- [ ] README reverted to near-original; only delta this PR directly requires
- [ ] Regression test written; fixture committed
- [ ] `schemas.py` → `rule_based_schema.py`
- [ ] `experiment.py` → `mvp_runner.py`
- [ ] `config/mvp.yaml` updated: multiplier to `[1,1,1,1,1]`, agents block added
- [ ] `config/mvp_with_mask.yaml` added

### Follow-up PRs (not blockers)
- [ ] Cross-reference validators (§4.2)
- [ ] Schema-reference mode stub (§4.3)
- [ ] Transition registry (§5)
- [ ] Decouple `environment.py` from concrete transition import
- [ ] Remove enums from `rule_based_schema.py`; promote back to `schemas.py`

---

## 9. Open Questions

Two questions remain, both related to the future `heartsteps_learned.yaml` config.
Neither is a blocker for this PR.

**OQ-1 — `initial_state: default` semantics**  
In schema-reference mode, `default` presumably means "sample from the empirical
initial-state distribution". Should `default` be a reserved keyword validated
explicitly, or should schema-reference mode skip `initial_state` validation entirely?
Resolving this when the learned transition type is implemented is fine.

**OQ-2 — Schema registry resolution**  
When `states: {schema: heartsteps}` is encountered, where does the resolver look?
Options: (a) a `config/schemas/heartsteps.yaml` file, (b) a Python module
`rl_health_interventions.schemas.heartsteps`, (c) a registry dict mirroring the
transition registry. The stub `NotImplementedError` in this PR just needs to
name the right future direction — which do you prefer?
