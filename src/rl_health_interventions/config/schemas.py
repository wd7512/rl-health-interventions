from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, RootModel, model_validator

_KNOWN_TRANSITION_TYPES = frozenset({"rule_based", "random", "bootstrap"})


class TransitionProbabilities(RootModel):
    root: dict[str, dict[str, dict[str, float]]]

    @model_validator(mode="after")
    def _check_probabilities(self) -> TransitionProbabilities:
        for state, actions in self.root.items():
            if not actions:
                raise ValueError(
                    f"Transition probabilities for state '{state}' has no actions"
                )
            for action, probs in actions.items():
                if not probs:
                    raise ValueError(
                        f"Transition probabilities for ({state}, {action}) is empty"
                    )
                for target, p in probs.items():
                    if p < 0.0:
                        raise ValueError(
                            f"Probability for {state} -> {target} under {action} "
                            f"cannot be negative: {p}"
                        )
                _epsilon = 1e-6
                total = sum(probs.values())
                if abs(total - 1.0) > _epsilon:
                    raise ValueError(
                        f"Probabilities for ({state}, {action}) sum to {total}, expected 1.0"
                    )
        return self


class CyclicAdvance(BaseModel):
    type: Literal["cyclic"] = "cyclic"
    granularity: Literal["daily"] = "daily"
    pattern: Annotated[list[str], Field(min_length=1)]


class Condition(BaseModel):
    factor: str
    type: Literal["in"]
    values: list[str]


class RollingWindowCountAdvance(BaseModel):
    type: Literal["rolling_window_count"] = "rolling_window_count"
    window_size: Annotated[int, Field(gt=0)] = 3
    conditions: list[Condition]
    mapping: dict[int, str]


class FactorConfig(BaseModel):
    names: list[str]
    advanced: CyclicAdvance | RollingWindowCountAdvance | None = None


class StateConfig(BaseModel):
    variables: dict[str, FactorConfig] = {}


class RewardVariable(BaseModel):
    source: str
    mapping: dict[str, float]


class RewardConfig(BaseModel):
    constants: dict[str, float] = {}
    variables: dict[str, RewardVariable]
    formula: str


class TransitionModelConfig(BaseModel):
    type: str = "rule_based"
    transition_probabilities: TransitionProbabilities | None = None
    table_dir: str | None = None
    table: str | None = None


_RESERVED_STATE_NAMES = frozenset(
    {"day", "step_of_day", "steps_per_day", "global_step", "factor_values", "state_key"}
)

_KNOWN_AGENT_TYPES = frozenset(
    {
        "thompson_sampling",
        "random",
        "epsilon_greedy",
        "ucb",
        "decaying_epsilon_greedy",
        "fixed",
        "comb_weighted_fixed",
        "heartsteps",
        "q_learning",
        "dqn",
        "reinforce",
        "ppo",
    }
)


def _check_name(fname: str, field: str, val: str, names: set[str]) -> None:
    if val not in names:
        raise ValueError(
            f"state.variables.{fname}.advanced.{field} value "
            f"'{val}' not in variable names: {sorted(names)}"
        )


def _require_positive(
    value: float | None, field: str, agent_type: str, *, required: bool = False
) -> None:
    if value is None:
        if required:
            raise ValueError(f"{field} must be provided for {agent_type}")
        return
    if value <= 0:
        raise ValueError(f"{field} must be > 0 for {agent_type}")


def _require_in_range(
    value: float | None, lo: float, hi: float, field: str, agent_type: str
) -> None:
    if value is not None and not (lo <= value <= hi):
        raise ValueError(f"{field} must be in [{lo}, {hi}] for {agent_type}")


def _require_in_range_open_lo(
    value: float | None, lo: float, hi: float, field: str, agent_type: str
) -> None:
    if value is not None and not (lo < value <= hi):
        raise ValueError(f"{field} must be in ({lo}, {hi}] for {agent_type}")


def _reject_fields(config: AgentConfig, fields: list[str], agent_type: str) -> None:
    """Raise ValueError if any of the given fields are set but not applicable for agent type."""
    violators = [f for f in fields if getattr(config, f) is not None]
    if violators:
        if len(violators) == 1:
            raise ValueError(
                f"field {violators[0]} is not applicable for agent type {agent_type}"
            )
        raise ValueError(
            f"fields {', '.join(sorted(violators))} are not applicable for agent type {agent_type}"
        )


class AgentConfig(BaseModel):
    type: str
    alpha_prior: float | None = None
    beta_prior: float | None = None
    epsilon: float | None = None
    epsilon_start: float | None = None
    epsilon_min: float | None = None
    decay_steps: int | None = None
    c: float | None = None
    contextual: bool = False
    context_features: str | list[str] | None = None
    action: str | None = None
    gamma: float | None = None
    lr: float | None = None
    hidden_dim: int | list[int] | None = None
    policy_hidden_dim: int | list[int] | None = None
    value_hidden_dim: int | list[int] | None = None
    state_dim: int | None = None
    grad_clip: float | None = None
    batch_size: int | None = None
    buffer_size: int | None = None
    target_update_freq: int | None = None
    clip_eps: float | None = None
    gae_lambda: float | None = None
    ppo_epochs: int | None = None
    lambda_dosage: float | None = None
    w: float | None = None
    epsilon_0: float | None = None
    epsilon_1: float | None = None
    sigma_sq: float | None = None
    reference_action: str | None = None
    prior_mean: float | None = None
    prior_cov: float | None = None
    f_features: str | list[str] | None = None
    g_features: str | list[str] | None = None
    persona_comb_file: str | None = None
    persona_name: str | None = None
    time_preference: str | None = None

    @model_validator(mode="after")
    def _validate_agent(self) -> AgentConfig:
        if self.type not in _KNOWN_AGENT_TYPES:
            raise ValueError(f"Unknown agent type: {self.type}")
        if self.type == "fixed":
            if self.action is None:
                raise ValueError("action must be provided for fixed agent")
            if not isinstance(self.action, str) or not self.action.strip():
                raise ValueError("action must be a non-empty string for fixed agent")
            if (
                self.alpha_prior is not None
                or self.beta_prior is not None
                or self.epsilon is not None
                or self.epsilon_start is not None
                or self.c is not None
                or self.decay_steps is not None
                or self.contextual
            ):
                raise ValueError(
                    "fixed agent does not accept learning hyperparameters or contextual"
                )
            if (
                self.persona_comb_file is not None
                or self.persona_name is not None
                or self.time_preference is not None
            ):
                raise ValueError(
                    "fixed agent does not accept persona_comb_file, persona_name, or time_preference"
                )
            return self
        if self.type == "comb_weighted_fixed":
            if self.persona_comb_file is None or not self.persona_comb_file.strip():
                raise ValueError(
                    "persona_comb_file must be provided for comb_weighted_fixed"
                )
            if self.persona_name is None or not self.persona_name.strip():
                raise ValueError("persona_name must be provided for comb_weighted_fixed")
            if (
                self.time_preference is not None
                and self.time_preference not in {"morning", "afternoon", "no_preference"}
            ):
                raise ValueError(
                    "time_preference must be one of morning, afternoon, no_preference"
                )
            if (
                self.action is not None
                or self.alpha_prior is not None
                or self.beta_prior is not None
                or self.epsilon is not None
                or self.epsilon_start is not None
                or self.c is not None
                or self.decay_steps is not None
                or self.contextual
            ):
                raise ValueError(
                    "comb_weighted_fixed does not accept action, learning hyperparameters, or contextual"
                )
            return self
        if self.contextual:
            if self.type not in (
                "thompson_sampling",
                "epsilon_greedy",
                "ucb",
                "decaying_epsilon_greedy",
            ):
                raise ValueError(
                    f"contextual=True is only supported for thompson_sampling, "
                    f"epsilon_greedy, ucb, and decaying_epsilon_greedy, got {self.type}"
                )
            if self.context_features is None:
                raise ValueError(
                    "context_features must be provided when contextual=True"
                )
            if (
                isinstance(self.context_features, str)
                and not self.context_features.strip()
            ):
                raise ValueError(
                    "context_features must be a non-empty string when contextual=True"
                )
            if isinstance(self.context_features, list):
                if not self.context_features:
                    raise ValueError(
                        "context_features must be a non-empty list when contextual=True"
                    )
                if not all(
                    isinstance(f, str) and f.strip() for f in self.context_features
                ):
                    raise ValueError(
                        "context_features list elements must be non-empty strings"
                    )
        elif self.context_features is not None:
            raise ValueError(
                "context_features must not be provided when contextual=False"
            )
        if self.type == "thompson_sampling":
            if self.alpha_prior is None or self.alpha_prior <= 0:
                raise ValueError("alpha_prior must be > 0 for thompson_sampling")
            if self.beta_prior is None or self.beta_prior <= 0:
                raise ValueError("beta_prior must be > 0 for thompson_sampling")
            if self.epsilon is not None or self.c is not None:
                raise ValueError("thompson_sampling agent does not accept epsilon or c")
        if self.type == "epsilon_greedy":
            if self.epsilon is None or not (0 <= self.epsilon <= 1):
                raise ValueError("epsilon must be in [0, 1] for epsilon_greedy")
            if (
                self.alpha_prior is not None
                or self.beta_prior is not None
                or self.c is not None
            ):
                raise ValueError(
                    "epsilon_greedy agent does not accept alpha_prior, beta_prior, or c"
                )
        if self.type == "ucb":
            if self.c is None or self.c <= 0:
                raise ValueError("c must be > 0 for ucb")
            if (
                self.alpha_prior is not None
                or self.beta_prior is not None
                or self.epsilon is not None
            ):
                raise ValueError(
                    "ucb agent does not accept alpha_prior, beta_prior, or epsilon"
                )
        if self.type == "random" and (
            self.alpha_prior is not None
            or self.beta_prior is not None
            or self.epsilon is not None
            or self.c is not None
        ):
            raise ValueError("random agent does not accept any hyperparameters")
        if self.type == "decaying_epsilon_greedy":
            if self.epsilon_start is None or not (0 <= self.epsilon_start <= 1):
                raise ValueError(
                    "epsilon_start must be in [0, 1] for decaying_epsilon_greedy"
                )
            if self.epsilon_min is not None and not (0 <= self.epsilon_min <= 1):
                raise ValueError(
                    "epsilon_min must be in [0, 1] for decaying_epsilon_greedy"
                )
            if (
                self.epsilon_min is not None
                and self.epsilon_start is not None
                and self.epsilon_min > self.epsilon_start
            ):
                raise ValueError(
                    "epsilon_min must not exceed epsilon_start for decaying_epsilon_greedy"
                )
            if self.decay_steps is not None and self.decay_steps <= 0:
                raise ValueError("decay_steps must be > 0 for decaying_epsilon_greedy")
            if (
                self.alpha_prior is not None
                or self.beta_prior is not None
                or self.epsilon is not None
                or self.c is not None
            ):
                raise ValueError(
                    "decaying_epsilon_greedy agent does not accept alpha_prior, beta_prior, epsilon, or c"
                )
        if self.type == "heartsteps":
            if self.gamma is not None and not (0 < self.gamma <= 1):
                raise ValueError("gamma must be in (0, 1] for heartsteps")
            if self.lambda_dosage is not None and not (0 < self.lambda_dosage < 1):
                raise ValueError("lambda_dosage must be in (0, 1) for heartsteps")
            if self.w is not None and not (0 <= self.w <= 1):
                raise ValueError("w must be in [0, 1] for heartsteps")
            if self.sigma_sq is not None and self.sigma_sq <= 0:
                raise ValueError("sigma_sq must be > 0 for heartsteps")
            if self.prior_mean is not None and not isinstance(
                self.prior_mean, (int, float)
            ):
                raise ValueError("prior_mean must be a number for heartsteps")
            if self.prior_cov is not None and self.prior_cov <= 0:
                raise ValueError("prior_cov must be > 0 for heartsteps")
            if (
                self.alpha_prior is not None
                or self.beta_prior is not None
                or self.epsilon is not None
                or self.epsilon_start is not None
                or self.c is not None
                or self.decay_steps is not None
                or self.epsilon_0 is not None
                or self.epsilon_1 is not None
            ):
                raise ValueError(
                    "heartsteps agent does not accept alpha_prior, beta_prior, epsilon, epsilon_start, c, decay_steps, epsilon_0, or epsilon_1"
                )
        # ── Q-Learning ──────────────────────────────────────────────
        if self.type == "q_learning":
            _require_positive(self.lr, "lr", self.type, required=True)
            _require_in_range(self.gamma, 0.0, 1.0, "gamma", self.type)
            _require_in_range(self.epsilon, 0.0, 1.0, "epsilon", self.type)
            _reject_fields(
                self,
                [
                    "state_dim",
                    "hidden_dim",
                    "policy_hidden_dim",
                    "value_hidden_dim",
                    "batch_size",
                    "buffer_size",
                    "target_update_freq",
                    "clip_eps",
                    "gae_lambda",
                    "ppo_epochs",
                ],
                "q_learning",
            )
        # ── DQN ────────────────────────────────────────────────────
        if self.type == "dqn":
            _require_positive(self.lr, "lr", self.type, required=True)
            _require_in_range(self.gamma, 0.0, 1.0, "gamma", self.type)
            _require_in_range(self.epsilon, 0.0, 1.0, "epsilon", self.type)
            _require_positive(self.state_dim, "state_dim", self.type)
            _require_positive(self.batch_size, "batch_size", "dqn")
            _require_positive(self.buffer_size, "buffer_size", "dqn")
            _require_positive(self.decay_steps, "decay_steps", "dqn")
            _require_positive(self.target_update_freq, "target_update_freq", "dqn")
            if self.grad_clip is not None and self.grad_clip < 0:
                raise ValueError("grad_clip must be non-negative for dqn")
            if (
                self.buffer_size is not None
                and self.batch_size is not None
                and self.buffer_size < self.batch_size
            ):
                raise ValueError("buffer_size must be >= batch_size for dqn")
            _require_in_range(self.epsilon_start, 0.0, 1.0, "epsilon_start", "dqn")
            _require_in_range(self.epsilon_min, 0.0, 1.0, "epsilon_min", "dqn")
            if (
                self.epsilon_start is not None
                and self.epsilon_min is not None
                and self.epsilon_min > self.epsilon_start
            ):
                raise ValueError("epsilon_min must not exceed epsilon_start for dqn")
            if self.hidden_dim is not None:
                if isinstance(self.hidden_dim, int):
                    _require_positive(self.hidden_dim, "hidden_dim", "dqn")
                elif not self.hidden_dim or not all(
                    isinstance(h, int) and h > 0 for h in self.hidden_dim
                ):
                    raise ValueError("hidden_dim list must contain positive integers")
            _reject_fields(
                self,
                [
                    "policy_hidden_dim",
                    "value_hidden_dim",
                    "clip_eps",
                    "gae_lambda",
                    "ppo_epochs",
                ],
                "dqn",
            )
        # ── REINFORCE ──────────────────────────────────────────────
        if self.type == "reinforce":
            _require_positive(self.lr, "lr", self.type, required=True)
            _require_in_range_open_lo(self.gamma, 0.0, 1.0, "gamma", self.type)
            _require_positive(self.state_dim, "state_dim", self.type)
            _reject_fields(
                self,
                [
                    "policy_hidden_dim",
                    "value_hidden_dim",
                    "batch_size",
                    "buffer_size",
                    "target_update_freq",
                    "clip_eps",
                    "gae_lambda",
                    "ppo_epochs",
                ],
                "reinforce",
            )
        # ── PPO ────────────────────────────────────────────────────
        if self.type == "ppo":
            _require_positive(self.lr, "lr", self.type, required=True)
            _require_in_range_open_lo(self.gamma, 0.0, 1.0, "gamma", self.type)
            _require_positive(self.state_dim, "state_dim", self.type)
            _require_in_range(self.gae_lambda, 0.0, 1.0, "gae_lambda", "ppo")
            _require_positive(self.clip_eps, "clip_eps", "ppo")
            _require_positive(self.ppo_epochs, "ppo_epochs", "ppo")
            _reject_fields(
                self,
                [
                    "batch_size",
                    "buffer_size",
                    "target_update_freq",
                ],
                "ppo",
            )
        return self


class MDPConfig(BaseModel):
    state: StateConfig
    initial_state: dict[str, str]
    actions: dict[str, dict] = {}
    reward: RewardConfig
    transition_model: TransitionModelConfig
    episode_days: int = Field(ge=1)
    steps_per_day: int = Field(ge=1)
    seed: int = 42
    reward_multiplier_by_step: list[float] | None = None
    agents: list[AgentConfig] = []

    @property
    def action_names(self) -> list[str]:
        return list(self.actions.keys())

    @model_validator(mode="before")
    @classmethod
    def _coerce_actions(cls, data: Any) -> Any:
        if isinstance(data, dict):
            actions = data.get("actions")
            if isinstance(actions, list):
                data = {**data, "actions": {a: {} for a in actions}}
        return data

    @model_validator(mode="after")
    def _validate_initial_state(self) -> MDPConfig:
        if not self.actions:
            raise ValueError("actions must be non-empty")
        variable_names = set(self.state.variables.keys())
        valid_keys = variable_names
        initial_keys = set(self.initial_state.keys())
        if not initial_keys:
            raise ValueError("initial_state must be non-empty")
        if not initial_keys.issubset(valid_keys):
            extra = initial_keys - valid_keys
            raise ValueError(
                f"initial_state keys {sorted(extra)} not found in "
                f"state.variables: {sorted(valid_keys)}"
            )
        for key, value in self.initial_state.items():
            cfg = self.state.variables[key]
            if value not in cfg.names:
                raise ValueError(
                    f"initial_state.{key}={value!r} not in variable {cfg.names}"
                )
        reserved = variable_names & _RESERVED_STATE_NAMES
        if reserved:
            raise ValueError(
                f"state variable names conflict with StateView reserved names: "
                f"{sorted(reserved)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_reward_variables(self) -> MDPConfig:
        for name, var_cfg in self.reward.variables.items():
            src = var_cfg.source
            if src == "action":
                for action_name in self.actions:
                    if action_name not in var_cfg.mapping:
                        raise ValueError(
                            f"reward.variables.{name} maps action "
                            f"'{action_name}' not in mapping"
                        )
            elif src.startswith("state."):
                factor_name = src.split(".", 1)[1]
                if factor_name not in self.state.variables:
                    raise ValueError(
                        f"reward.variables.{name} references unknown "
                        f"state variable '{factor_name}'"
                    )
                factor_cfg = self.state.variables[factor_name]
                unmapped = set(factor_cfg.names) - set(var_cfg.mapping.keys())
                if unmapped:
                    raise ValueError(
                        f"reward.variables.{name} missing mapping for "
                        f"state.variables.{factor_name} names: {sorted(unmapped)}"
                    )
            else:
                raise ValueError(
                    f"reward.variables.{name} source must be "
                    f"'action' or 'state.<name>', got '{src}'"
                )
        constant_names = set(self.reward.constants.keys())
        variable_names = set(self.reward.variables.keys())
        overlap = constant_names & variable_names
        if overlap:
            raise ValueError(
                f"reward constant and variable names conflict: {sorted(overlap)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_transition_model(self) -> MDPConfig:
        tm = self.transition_model
        if tm.type not in _KNOWN_TRANSITION_TYPES:
            raise ValueError(
                f"Unknown transition model type: {tm.type}. "
                f"Known: {sorted(_KNOWN_TRANSITION_TYPES)}"
            )
        if (
            tm.type == "rule_based"
            and tm.transition_probabilities is None
            and tm.table_dir is None
        ):
            raise ValueError(
                "rule_based transition requires either transition_probabilities or table_dir"
            )
        if tm.type == "random" and tm.transition_probabilities is not None:
            raise ValueError(
                "random transition does not accept transition_probabilities"
            )
        if tm.type == "bootstrap":
            if tm.table_dir is None:
                raise ValueError("bootstrap transition requires table_dir")
            if tm.transition_probabilities is not None:
                raise ValueError(
                    "bootstrap transition does not accept transition_probabilities"
                )
        tprobs = tm.transition_probabilities
        if tprobs is not None:
            if not self.state.variables:
                raise ValueError(
                    "transition_probabilities require at least one state variable"
                )
            first_var_name = next(iter(self.state.variables))
            first_var = self.state.variables[first_var_name]
            state_names = set(first_var.names)
            probs_root = tprobs.root
            if set(probs_root.keys()) != state_names:
                missing = state_names - set(probs_root.keys())
                extra = set(probs_root.keys()) - state_names
                parts = []
                if missing:
                    parts.append(
                        f"missing from transition_probabilities: {sorted(missing)}"
                    )
                if extra:
                    parts.append(
                        f"in transition_probabilities but not in {first_var_name} names: {sorted(extra)}"
                    )
                raise ValueError(
                    f"Transition probabilities must cover exactly the declared "
                    f"state variable '{first_var_name}' values — {'; '.join(parts)}"
                )
            if len(self.state.variables) > 1:
                raise ValueError(
                    "inline transition_probabilities only support one state variable; "
                    f"got {len(self.state.variables)}: {list(self.state.variables.keys())}"
                )
            for sv, actions_dict in probs_root.items():
                for action_name in self.action_names:
                    if action_name not in actions_dict:
                        raise ValueError(
                            f"Missing transition entry for ({sv}, {action_name})"
                        )
                for action_name, next_state_probs in actions_dict.items():
                    for target in next_state_probs:
                        if target not in state_names:
                            raise ValueError(
                                f"Transition target '{target}' not in {first_var_name} names"
                            )
                    if set(next_state_probs.keys()) != state_names:
                        raise ValueError(
                            f"Target distribution for ({sv}, {action_name}) "
                            f"must cover all {first_var_name} values"
                        )
        return self

    @model_validator(mode="after")
    def _validate_advanced_configs(self) -> MDPConfig:
        for fname, fcfg in self.state.variables.items():
            if fcfg.advanced is None:
                continue
            names_set = set(fcfg.names)
            if isinstance(fcfg.advanced, CyclicAdvance):
                for val in fcfg.advanced.pattern:
                    _check_name(fname, "pattern", val, names_set)
            elif isinstance(fcfg.advanced, RollingWindowCountAdvance):
                expected_keys = set(range(fcfg.advanced.window_size + 1))
                actual_keys = set(fcfg.advanced.mapping.keys())
                if actual_keys != expected_keys:
                    raise ValueError(
                        f"state.variables.{fname}.advanced.mapping keys "
                        f"{sorted(actual_keys)} must cover {sorted(expected_keys)}"
                    )
                for val in fcfg.advanced.mapping.values():
                    _check_name(fname, "mapping", val, names_set)
                for cond in fcfg.advanced.conditions:
                    if (
                        cond.factor not in self.state.variables
                        and cond.factor != "action"
                    ):
                        raise ValueError(
                            f"state.variables.{fname}.advanced.conditions.factor "
                            f"'{cond.factor}' is not a declared variable or 'action'"
                        )
        return self

    @model_validator(mode="after")
    def _validate_reward_multiplier(self) -> MDPConfig:
        if (
            self.reward_multiplier_by_step is not None
            and len(self.reward_multiplier_by_step) != self.steps_per_day
        ):
            raise ValueError(
                f"reward_multiplier_by_step length "
                f"{len(self.reward_multiplier_by_step)} "
                f"must equal steps_per_day {self.steps_per_day}"
            )
        return self
