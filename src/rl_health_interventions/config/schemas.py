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


class TransitionModelConfig(BaseModel):
    type: str = "rule_based"
    transition_probabilities: TransitionProbabilities | None = None


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
        """
        Validate agent configuration constraints.

        Ensures the agent type is recognized, contextual mode is compatible with the agent type,
        context features are properly configured, and all hyperparameters are consistent with
        the selected agent type.

        Returns:
                AgentConfig: The validated agent configuration.
        """
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


class RewardWeightsConfig(BaseModel):
    mode: str = "multi_timescale"
    delayed_reward_interval: int = Field(ge=1, default=21)
    delayed_reward_value: float = Field(ge=0.0, default=10.0)
    delayed_reward_scale: float | None = Field(ge=0.0, default=None)
    delayed_reward_threshold: float | None = Field(ge=0.0, le=1.0, default=None)


class MDPConfig(BaseModel):
    episode_days: int = Field(ge=1)
    steps_per_day: int = Field(ge=1)
    seed: int = 42
    initial_state: str = "sedentary"
    states: Any
    actions: Any
    transition_model: TransitionModelConfig
    reward_multiplier_by_step: list[float] | None = None
    reward_weights: RewardWeightsConfig | None = None
    agents: list[AgentConfig] = []
    per_step_reward: list[dict[str, float]] | None = None

    @model_validator(mode="after")
    def _cross_reference_validators(self) -> MDPConfig:
        """Schema-ref mode: skip inline checks when states/actions have a schema key."""
        is_schema_ref = (isinstance(self.states, dict) and "schema" in self.states) or (
            isinstance(self.actions, dict) and "schema" in self.actions
        )
        if is_schema_ref:
            return self

        if not isinstance(self.states, dict):
            raise ValueError(
                "states must be a dictionary mapping state names to their configurations"
            )
        if not isinstance(self.actions, list):
            raise ValueError("actions must be a list of action names")

        state_names = set(self.states.keys())
        action_names = set(self.actions)

        if not self.states:
            raise ValueError("states must be non-empty")
        if self.initial_state not in state_names:
            raise ValueError(
                f"initial_state '{self.initial_state}' not in states: "
                f"{sorted(state_names)}"
            )
        for name, data in self.states.items():
            if not isinstance(data, dict) or "reward" not in data:
                raise ValueError(f"State '{name}' must have a numeric 'reward' field")
            if not isinstance(data["reward"], (int, float)):
                raise ValueError(
                    f"State '{name}' reward must be numeric, got {type(data['reward']).__name__}"
                )

        if not self.actions:
            raise ValueError("actions must be non-empty")
        if len(action_names) != len(self.actions):
            raise ValueError("actions contain duplicates")

        if (
            self.transition_model.type == "rule_based"
            and self.transition_model.transition_probabilities is None
        ):
            raise ValueError(
                "transition_probabilities must be provided for rule_based transition model"
            )

        tprobs = self.transition_model.transition_probabilities
        if tprobs is not None:
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
                        f"in transition_probabilities but not in states: {sorted(extra)}"
                    )
                raise ValueError(
                    f"Transition probabilities must cover exactly the declared states — {'; '.join(parts)}"
                )

            for state, actions_dict in probs_root.items():
                for action in self.actions:
                    if action not in actions_dict:
                        raise ValueError(
                            f"Missing transition entry for ({state}, {action})"
                        )

            for state, actions_dict in probs_root.items():
                for action, next_state_probs in actions_dict.items():
                    for target in next_state_probs:
                        if target not in state_names:
                            raise ValueError(
                                f"Transition target '{target}' not in states"
                            )
                    if set(next_state_probs.keys()) != state_names:
                        raise ValueError(
                            f"Target distribution for ({state}, {action}) "
                            f"must cover all states"
                        )

        return self

    @model_validator(mode="after")
    def _validate_reward_multiplier(self) -> MDPConfig:
        if self.reward_multiplier_by_step is not None:
            if len(self.reward_multiplier_by_step) != self.steps_per_day:
                raise ValueError(
                    f"reward_multiplier_by_step length {len(self.reward_multiplier_by_step)} "
                    f"must equal steps_per_day {self.steps_per_day}"
                )
        return self

    @model_validator(mode="after")
    def _compute_per_step_reward(self) -> MDPConfig:
        if isinstance(self.states, dict) and "schema" in self.states:
            return self
        multiplier = self.reward_multiplier_by_step or [1.0] * self.steps_per_day
        self.per_step_reward = [
            {state: data["reward"] * mult for state, data in self.states.items()}
            for mult in multiplier
        ]
        return self
