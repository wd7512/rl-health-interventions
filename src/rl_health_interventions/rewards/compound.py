from __future__ import annotations

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.rewards._base import RewardHandler
from rl_health_interventions.state import StateView


class CompoundReward(RewardHandler):
    """Reward computed from config.reward.values and per_step_multiplier.

    In sprint1 factored config, reward is computed as:
        reward = alpha * f(step_bin') + (1-alpha) * g(sleep') - lambda * I[a != idle]

    For schema-ref configs, the reward must be loaded externally.
    """

    def __init__(self, config: MDPConfig) -> None:
        if config.state.schema is not None:
            raise NotImplementedError(
                "Schema-ref configs (state: {schema: ...}) are not yet supported. "
                "Use inline state factor definitions with rule_based transition/reward. "
                "See docs/ROADMAP.md for Phase 2 plans."
            )
        if config.state.factors is None:
            raise ValueError("state.factors must be provided for compound reward")
        self._reward_factor = config.reward.factor
        self._reward_values = config.reward.values
        self._per_step_mult = (
            config.reward.per_step_multiplier or [1.0] * config.steps_per_day
        )
        self._action_penalty_mult = config.reward.action_penalty_multiplier
        self._action_configs = config.actions
        self._alpha = (
            config.reward.constants.get("alpha", 1.0)
            if config.reward.constants
            else 1.0
        )
        self._sleep_factor = config.reward.sleep_factor
        self._sleep_values = config.reward.sleep_values or {}

    def reward(
        self, state: StateView, action: str, step_idx: int
    ) -> tuple[float, bool]:
        step_bin_value = self._reward_values.get(
            getattr(state, self._reward_factor), 0.0
        )
        sleep_value = (
            self._sleep_values.get(getattr(state, self._sleep_factor), 0.0)
            if self._sleep_factor
            else 0.0
        )
        step_mult = (
            self._per_step_mult[step_idx]
            if step_idx < len(self._per_step_mult)
            else 1.0
        )
        action_penalty = 0.0
        if action in self._action_configs:
            raw = self._action_configs[action]
            if hasattr(raw, "action_penalty"):
                action_penalty = raw.action_penalty  # type: ignore[union-attr]
            elif isinstance(raw, dict):
                action_penalty = raw.get("action_penalty", 0.0)
        penalty = action_penalty * self._action_penalty_mult
        r = self._alpha * step_bin_value + (1 - self._alpha) * sleep_value
        return r * step_mult - penalty, False


def register() -> None:
    from rl_health_interventions.rewards import REGISTRY

    REGISTRY["compound"] = CompoundReward
