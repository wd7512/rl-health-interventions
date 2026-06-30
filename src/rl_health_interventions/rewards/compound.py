from __future__ import annotations

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.rewards._base import RewardHandler
from rl_health_interventions.state import StateView


class CompoundReward(RewardHandler):
    """Reward computed from config.reward.values and per_step_multiplier.

    In the new factored config, reward is computed as:
        reward = reward_values[state.reward_factor] * per_step_multiplier[step_idx]
                 - action_penalty * action_penalty_multiplier

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

    def reward(
        self, state: StateView, action: str, step_idx: int
    ) -> tuple[float, bool]:
        factor_value = getattr(state, self._reward_factor)
        base_reward = self._reward_values.get(factor_value, 0.0)
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
        return base_reward * step_mult - penalty, False


def register() -> None:
    from rl_health_interventions.rewards import REGISTRY

    REGISTRY["compound"] = CompoundReward
