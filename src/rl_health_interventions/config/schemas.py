from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, RootModel, model_validator


class TransitionProbabilities(RootModel):
    root: dict[str, dict[str, dict[str, float]]]

    @model_validator(mode="after")
    def _check_probabilities(self) -> TransitionProbabilities:
        for state, actions in self.root.items():
            for action, probs in actions.items():
                for target, p in probs.items():
                    if p < 0.0:
                        raise ValueError(
                            f"Probability for {state} -> {target} under {action} "
                            f"cannot be negative: {p}"
                        )
                total = sum(probs.values())
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(
                        f"Probabilities for ({state}, {action}) sum to {total}, expected 1.0"
                    )
        return self


_KNOWN_TRANSITION_TYPES = frozenset({"rule_based", "random", "learned", "bootstrap"})
_TABLE_BACKED_TRANSITION_TYPES = frozenset({"rule_based", "random", "bootstrap"})


class FactorConfig(BaseModel):
    dims: int = Field(ge=1)
    names: list[str]
    boundaries: list[float] | None = None

    @model_validator(mode="after")
    def _check_names_length(self) -> FactorConfig:
        if len(self.names) != self.dims:
            raise ValueError(
                f"names length ({len(self.names)}) must equal dims ({self.dims})"
            )
        return self

    @model_validator(mode="after")
    def _check_names_unique(self) -> FactorConfig:
        if len(set(self.names)) != len(self.names):
            raise ValueError(
                f"factor names must be unique, got duplicates: {self.names}"
            )
        return self

    @model_validator(mode="after")
    def _check_boundaries(self) -> FactorConfig:
        if self.boundaries is not None:
            expected = self.dims - 1
            if len(self.boundaries) != expected:
                raise ValueError(
                    f"boundaries length ({len(self.boundaries)}) "
                    f"must equal dims - 1 ({expected})"
                )
        return self


class ActionConfig(BaseModel):
    action_penalty: float = 0.0


class TransitionModelConfig(BaseModel):
    type: str = "rule_based"
    table_dir: str | None = None
    table: str | None = None

    @model_validator(mode="after")
    def _validate_config(self) -> TransitionModelConfig:
        if self.type not in _KNOWN_TRANSITION_TYPES:
            raise ValueError(
                f"Unknown transition type: '{self.type}'. "
                f"Supported types: {sorted(_KNOWN_TRANSITION_TYPES)}"
            )
        if self.type in _TABLE_BACKED_TRANSITION_TYPES and self.table_dir is None:
            raise ValueError(f"table_dir is required for transition type '{self.type}'")
        return self


class RewardConfig(BaseModel):
    factor: str
    values: dict[str, float]
    action_penalty_multiplier: float = 0.0
    per_step_multiplier: list[float] | None = None


class StateConfig(BaseModel):
    factors: dict[str, FactorConfig] | None = None
    schema: str | None = None

    @model_validator(mode="after")
    def _check_exactly_one(self) -> StateConfig:
        has_factors = self.factors is not None
        has_schema = self.schema is not None
        if has_factors == has_schema:
            if has_factors:
                raise ValueError(
                    "exactly one of 'factors' or 'schema' must be set, not both"
                )
            else:
                raise ValueError("exactly one of 'factors' or 'schema' must be set")
        return self


_KNOWN_AGENT_TYPES = frozenset(
    {"thompson_sampling", "random", "epsilon_greedy", "ucb", "decaying_epsilon_greedy"}
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
    context_feature: str | None = None

    @model_validator(mode="after")
    def _validate_agent(self) -> AgentConfig:
        if self.type not in _KNOWN_AGENT_TYPES:
            raise ValueError(f"Unknown agent type: {self.type}")
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
            if self.context_feature is None or not self.context_feature.strip():
                raise ValueError(
                    "context_feature must be a non-empty string when contextual=True"
                )
        else:
            if self.context_feature is not None:
                raise ValueError(
                    "context_feature must not be provided when contextual=False"
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
        if self.type == "random":
            if (
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
        return self


class MDPConfig(BaseModel):
    episode_days: int = Field(ge=1)
    steps_per_day: int = Field(ge=1)
    seed: int = 42
    initial_state: dict[str, str]
    state: StateConfig
    actions: dict[str, Any]
    reward: RewardConfig
    transition_model: TransitionModelConfig
    agents: list[AgentConfig] = []

    @model_validator(mode="before")
    @classmethod
    def _preprocess_actions(cls, data: dict) -> dict:
        if isinstance(data.get("actions"), list):
            data["actions"] = {a: {} for a in data["actions"]}
        return data

    @model_validator(mode="after")
    def _coerce_actions(self) -> MDPConfig:
        if self.state.schema is not None:
            return self
        coerced: dict[str, Any] = {}
        for name, cfg in self.actions.items():
            if isinstance(cfg, ActionConfig):
                coerced[name] = cfg
            elif isinstance(cfg, dict):
                coerced[name] = ActionConfig(**cfg)
            else:
                raise ValueError(
                    f"Invalid action config for '{name}': expected dict or "
                    f"ActionConfig, got {type(cfg).__name__}"
                )
        self.actions = coerced
        return self

    @model_validator(mode="after")
    def _cross_reference_validators(self) -> MDPConfig:
        if self.state.schema is not None:
            return self

        factor_configs = self.state.factors
        if factor_configs is None:
            raise ValueError(
                "state.factors must be provided when state.schema is not set"
            )

        factor_names = set(factor_configs.keys())

        initial_keys = set(self.initial_state.keys())
        if initial_keys != factor_names:
            missing = factor_names - initial_keys
            extra = initial_keys - factor_names
            parts = []
            if missing:
                parts.append(f"missing factors: {sorted(missing)}")
            if extra:
                parts.append(f"extra keys: {sorted(extra)}")
            raise ValueError(
                f"initial_state keys must match factor names — {'; '.join(parts)}"
            )

        for factor_name, factor_value in self.initial_state.items():
            fc = factor_configs[factor_name]
            if factor_value not in fc.names:
                raise ValueError(
                    f"initial_state['{factor_name}'] = '{factor_value}' is not a valid"
                    f" value for factor '{factor_name}'. Valid values: {fc.names}"
                )

        if self.reward.factor not in factor_configs:
            raise ValueError(
                f"reward.factor '{self.reward.factor}' not found in state.factors. "
                f"Available factors: {sorted(factor_names)}"
            )

        reward_factor = self.reward.factor
        expected_keys = set(factor_configs[reward_factor].names)
        actual_keys = set(self.reward.values.keys())
        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            parts = []
            if missing:
                parts.append(f"missing keys: {sorted(missing)}")
            if extra:
                parts.append(f"extra keys: {sorted(extra)}")
            raise ValueError(
                f"reward.values keys must match factor '{reward_factor}' names — "
                f"{'; '.join(parts)}"
            )

        if self.reward.per_step_multiplier is not None:
            if len(self.reward.per_step_multiplier) != self.steps_per_day:
                raise ValueError(
                    f"reward.per_step_multiplier length "
                    f"({len(self.reward.per_step_multiplier)}) "
                    f"must equal steps_per_day ({self.steps_per_day})"
                )

        return self

    @property
    def action_names(self) -> list[str]:
        keys = list(self.actions.keys())
        if self.state.schema is not None:
            return [k for k in keys if k != "schema"]
        return keys

    @property
    def factor_configs(self) -> dict[str, FactorConfig]:
        if self.state.factors is not None:
            return self.state.factors
        return {}
