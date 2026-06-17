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


_KNOWN_AGENT_TYPES = frozenset({"thompson_sampling", "random", "epsilon_greedy", "ucb"})


class AgentConfig(BaseModel):
    type: str
    alpha_prior: float | None = None
    beta_prior: float | None = None
    epsilon: float | None = None
    c: float | None = None

    @model_validator(mode="after")
    def _validate_agent(self) -> AgentConfig:
        if self.type not in _KNOWN_AGENT_TYPES:
            raise ValueError(f"Unknown agent type: {self.type}")
        if self.type == "thompson_sampling":
            if self.alpha_prior is None or self.alpha_prior <= 0:
                raise ValueError("alpha_prior must be > 0 for thompson_sampling")
            if self.beta_prior is None or self.beta_prior <= 0:
                raise ValueError("beta_prior must be > 0 for thompson_sampling")
        if self.type == "epsilon_greedy":
            if self.epsilon is None or not (0 < self.epsilon <= 1):
                raise ValueError("epsilon must be in (0, 1] for epsilon_greedy")
        if self.type == "ucb":
            if self.c is None or self.c <= 0:
                raise ValueError("c must be > 0 for ucb")
        return self


class MDPConfig(BaseModel):
    episode_days: int = Field(ge=1)
    steps_per_day: int = Field(ge=1)
    seed: int = 42
    initial_state: str = "sedentary"
    states: Any
    actions: Any
    transition_model: TransitionModelConfig
    reward_multiplier_by_step: list[float] | None = None
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

        state_names = {k for k, v in self.states.items()}
        action_names = set(self.actions)

        if not self.states:
            raise ValueError("states must be non-empty")
        for name, data in self.states.items():
            if not isinstance(data, dict) or "reward" not in data:
                raise ValueError(f"State '{name}' must have a numeric 'reward' field")

        if not self.actions:
            raise ValueError("actions must be non-empty")
        if len(action_names) != len(self.actions):
            raise ValueError("actions contain duplicates")

        tprobs = self.transition_model.transition_probabilities
        if tprobs is not None:
            probs_root = tprobs.root
            for state in probs_root:
                if state not in state_names:
                    raise ValueError(f"Transition state '{state}' not in states")

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
