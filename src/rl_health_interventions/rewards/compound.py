from __future__ import annotations

from rl_health_interventions.config.schemas import MDPConfig, RewardWeightsConfig
from rl_health_interventions.rewards._base import RewardHandler
from rl_health_interventions.state import StateView


class CompoundReward(RewardHandler):
    def __init__(self, config: MDPConfig) -> None:
        p = config.per_step_reward
        if p is None:
            raise NotImplementedError(
                "Schema-ref configs (states: {schema: ...}) are not yet supported. "
                "Use inline state definitions with rule_based transition/reward. "
                "See docs/ROADMAP.md for Phase 2 plans."
            )
        self._per_step_reward = p
        self._reward_weights: RewardWeightsConfig | None = config.reward_weights
        self._active_count: int = 0

    def reward(
        self, state: StateView, action: str, step_idx: int
    ) -> tuple[float, bool]:
        base = self._per_step_reward[step_idx][state.activity]

        if self._reward_weights is None:
            # Simple mode: no delayed reward
            return base, False

        delayed_interval = self._reward_weights.delayed_reward_interval
        global_step = state.global_step

        if self._reward_weights.delayed_reward_scale is not None:
            # Option B (scaled)
            if state.activity == "active":
                self._active_count += 1

            if global_step > 0 and global_step % delayed_interval == 0:
                rate = self._active_count / delayed_interval
                threshold = (
                    self._reward_weights.delayed_reward_threshold
                    if self._reward_weights.delayed_reward_threshold is not None
                    else 0.0
                )
                bonus = (
                    self._reward_weights.delayed_reward_scale * rate
                    if rate >= threshold
                    else 0.0
                )
                self._active_count = 0
            else:
                bonus = 0.0
        else:
            # Option A (flat)
            if global_step > 0 and global_step % delayed_interval == 0:
                bonus = self._reward_weights.delayed_reward_value
            else:
                bonus = 0.0

        return base + bonus, False

    def reset(self) -> None:
        self._active_count = 0


def register() -> None:
    from rl_health_interventions.rewards import REGISTRY

    REGISTRY["compound"] = CompoundReward
