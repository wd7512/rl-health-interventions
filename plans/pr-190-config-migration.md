# PR 1 (#190) — Config Schema Migration + FixedAgent

## Scope

Full config schema migration from legacy flat `states`/`actions` list format to unified
`state.variables + state.parameters` / formula reward format. Includes `FactoredState`
refactor, safe expression parser, `FixedAgent`, and migration of every YAML, runner script,
and test fixture. End state: one config format, one model, zero legacy paths.

## Key decisions (locked before implementation)

1. **Config format**: `state.variables` (stochastic, LLM-bootstrapped) + `state.parameters`
   (deterministic, config-driven `advance` formulas). No more top-level `states` dict or
   `actions` list. Single unified `MDPConfig` model — no sibling types, no union return.
2. **Reward**: Expression parser (`constants` + `variables` + `formula`), ast-based safe
   evaluator, no `eval()`. `ExpressionReward` is the only reward handler — `CompoundReward`
   is removed.
3. **Actions**: Dict of `{action_name: {}}`. List shorthand coerced to dict at the
   pydantic boundary. `ActionConfig` is empty — penalty lives in reward formula variables.
4. **`StateView` refactor**: Dict-backed immutable object with `__getattr__` for dynamic
   factor names. Replaces the frozen `activity`/`day`/`step_of_day` dataclass.
5. **Single factor name for MVP**: `activity_level`.
6. **FixedAgent**: Configurable fixed action (default `"idle"`). Replaces spec's
   non-existent `IdleOnlyAgent`.
7. **Contextual agents**: `context_feature` accepts `str | list[str]`; `_get_context_key`
   produces tuple keys for multi-field context.
8. **Transition probabilities**: Kept as an optional inline field alongside `table_dir` for
   table-backed transitions. MVP configs use inline probs, sprint1 uses JSON table files.
9. **MVP regression target**: 165.0 total reward — preserved because cache iteration order
   and RNG seeding are unchanged for inline-prob configs.
10. **All YAMLs migrated**: `config/rule_based.yaml`, `docs/sprint1/configs/sprint1.yaml`,
    and all 4 `docs/experimental_phases/mvp/configs/*.yaml` rewritten to the new format.

## File inventory

### Core source (10 modified, 2 new)

| File | Action |
|---|---|
| `config/schemas.py` | Replace `MDPConfig` with unified model — `state.variables` + `state.parameters`, formula reward, empty `ActionConfig`, multi-field `context_feature`, inline `transition_probabilities` kept optional |
| `config/loader.py` | Resolve `table_dir` relative to config path; remove dispatch (one model now) |
| `state.py` | Rewrite `StateView` as dict-backed with `__getattr__` factor resolution |
| `environment.py` | Update for new `StateView`; no sprint1 features |
| `episode.py` | Update `state.activity` → factor-based access |
| `rewards/__init__.py` | Remove `CompoundReward`; always return `ExpressionReward` |
| `rewards/compound.py` | Delete entirely |
| `agents/__init__.py` | Register `fixed` |
| `agents/contextual_bandits/_base.py` | Accept `list[str]` context_feature; tuple-key `_get_context_key` |
| `sweep.py` | `config.actions` → `config.action_names` |
| `__main__.py` | `config.actions` → `config.action_names` |
| **NEW** `rewards/expression.py` | `ExpressionParser` (AST safe evaluator) + `ExpressionReward` |
| **NEW** `agents/fixed.py` | `FixedAgent` — configurable fixed action, no learning |

### Config files (7 migrated)

| File | Action |
|---|---|
| `config/rule_based.yaml` | Rewrite to new format — factor `activity_level`, inline probs preserved |
| `docs/sprint1/configs/sprint1.yaml` | `state.factors` → `state.variables` + `state.parameters`. Add `agents` block with `context_feature: [step_bin, burden, day_of_week, sleep]` |
| `docs/experimental_phases/mvp/configs/mvp.yaml` | Rewrite |
| `docs/experimental_phases/mvp/configs/mvp_masked.yaml` | Rewrite |
| `docs/experimental_phases/mvp/configs/mvp_extensions.yaml` | Rewrite |
| `docs/experimental_phases/mvp/configs/mvp_extensions_masked.yaml` | Rewrite |

### Runner scripts (3 updated)

| File | Action |
|---|---|
| `docs/experimental_phases/mvp/optimal_bound.py` | Replace `config.states`/`config.actions` access with new model properties |
| `docs/experimental_phases/mvp/_shared.py` | `config.actions` → `config.action_names` |
| `docs/experimental_phases/mvp/run_experiments.py` | Verify no legacy field access (none expected) |

### Test files (12 updated, 4 new)

| File | Action |
|---|---|
| `conftest.py` | Rewrite fixtures with new `StateView` + new `MDPConfig` shape |
| `test_state.py` | Test dict-backed `StateView` |
| `test_environment.py` | Updated config construction |
| `test_optimal_bound.py` | All 7 `MDPConfig(...)` constructors rewritten to new shape |
| `test_mdp_config.py` | Updated YAML shapes |
| `test_mdp_validators.py` | Updated validation rules |
| `test_agents_registry.py` | Add `fixed` to known types |
| `test_contextual_bandit_base.py` | Add multi-field `context_feature` tests |
| `test_compound_reward.py` | Delete or repurpose |
| `test_dummy_step.py` | Updated config construction |
| `mvp_expected_rewards.json` | Re-base (should still be 165.0) |
| **NEW** `test_expression_parser.py` | Parse, compile, evaluate, reject unsafe ops |
| **NEW** `test_expression_reward.py` | Map state/action → variables, evaluate |
| **NEW** `test_fixed_agent.py` | Action selection, state ignored, update no-op |
| **NEW** `test_factored_config.py` | Unified `MDPConfig` validation — variables, parameters, formula, cross-reference |

## Design details

### Config model structure (`config/schemas.py`)

```python
class FactorConfig(BaseModel):
    dims: int = Field(ge=1)
    names: list[str]
    boundaries: list[float] | None = None

class ParameterConfig(BaseModel):
    advance: str  # Formula expression, evaluated at runtime (PR 2)

class StateConfig(BaseModel):
    variables: dict[str, FactorConfig]  # Stochastic, LLM-bootstrapped
    parameters: dict[str, ParameterConfig]  # Deterministic, formula-driven

class ActionConfig(BaseModel):
    pass  # Penalty lives in reward formula variables

class RewardVariable(BaseModel):
    source: str  # "state.<name>" or "action"
    mapping: dict[str, float]

class RewardConfig(BaseModel):
    constants: dict[str, float] = {}
    variables: dict[str, RewardVariable]
    formula: str

class TransitionModelConfig(BaseModel):
    type: str = "rule_based"
    transition_probabilities: TransitionProbabilities | None = None  # Inline (MVP)
    table_dir: str | None = None  # JSON table files (sprint1)
    table: str | None = None     # Specific table file within table_dir

class MDPConfig(BaseModel):  # ← Unified model, same name
    state: StateConfig
    initial_state: dict[str, str]
    actions: dict[str, ActionConfig]  # List coerced to dict in validator
    reward: RewardConfig
    transition_model: TransitionModelConfig
    episode_days: int = Field(ge=1)
    steps_per_day: int = Field(ge=1)
    seed: int = 42
    agents: list[AgentConfig] = []

    @property
    def action_names(self) -> list[str]: ...
```

Key validators:
- `@model_validator(mode="before")`: coerce `actions` list to dict
- `initial_state` keys = union of `variables` + `parameters` keys
- `initial_state` values are valid factor/parameter names
- `reward.variables[i].source` is valid: `state.<name>` with `name` in `variables`, or `"action"`
- `reward.constants` and `reward.variables` names don't conflict

### StateView (`state.py`)

```python
class StateView:
    """Immutable state with dynamic factor access via __getattr__."""
    def __init__(self, factors: dict[str, str], day: int = 0,
                 step_of_day: int = 0, steps_per_day: int = 5): ...

    def __getattr__(self, name: str) -> str:
        try:
            return self._factors[name]
        except KeyError:
            raise AttributeError(...)

    def __setattr__(self, name, value) -> None:
        raise AttributeError(f"StateView is immutable")

    # Properties: day, step_of_day, steps_per_day, global_step, state_key, factor_values
    # Methods: with_factors(**updates) -> StateView, with_advance(...) -> StateView
    # Equality, hashing, repr
```

### Expression parser (`rewards/expression.py`)

```python
ALLOWED_NODES = frozenset({
    ast.Expression, ast.Constant, ast.Name,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.UnaryOp, ast.USub,
})

class ExpressionParser:
    def __init__(self, formula: str):
        self._compiled = self._compile(formula)

    def evaluate(self, variables: dict[str, float]) -> float:
        ...

class ExpressionReward(RewardHandler):
    def __init__(self, config: MDPConfig): ...
    def reward(self, state: str | StateView, action: str,
               step_idx: int) -> tuple[float, bool]:
        # Resolve variables from state/action, call parser
        return value, False
```

### FixedAgent (`agents/fixed.py`)

```python
class FixedAgent(Agent):
    def __init__(self, action: str = "idle", **kwargs):
        self._action = action
    def select_action(self, state) -> str:
        return self._action
    # update() inherited from Agent (no-op)
```

## Not in scope (PRs 2–4)

- Sprint 1 environment loop (day boundary, burden update, day_of_week advance, sleep transition)
- `BootstrapTransition` model / JSON table loading
- Bootstrap script (Algorithm 2 — LLM → transition table)
- Evaluation runner (seeds, metrics, multi-agent comparison) — beyond what exists in MVP
- Contextual agent runtime behavior in sprint1 config (agents exist in YAML but aren't
  exercised until PR 2's environment)

## Verification

1. `uv run ruff format --check .`
2. `uv run ruff check`
3. `uv run ty check --exclude tests/`
4. `uv run pytest` — all tests pass, including:
   - `tests/test_regression_mvp.py` (165.0)
   - `tests/regression/test_mvp.py` (all 4 MVP configs against golden fixtures)
5. Manual: `load_config("config/rule_based.yaml")` — returns unified model, agents run
   episode, total reward matches fixture
6. Manual: `load_config("docs/sprint1/configs/sprint1.yaml")` — validates correctly
7. Manual: `uv run python docs/experimental_phases/mvp/optimal_bound.py mvp.yaml` — runs
   without error

## Amendment log

| Date | Change |
|------|--------|
| 2026-07-04 | Initial plan — grilling complete, decisions locked, ready to implement |


