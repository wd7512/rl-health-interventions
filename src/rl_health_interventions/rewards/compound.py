from __future__ import annotations

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.rewards._base import RewardHandler


class CompoundReward(RewardHandler):
    def __init__(self, config: MDPConfig) -> None:
        p = config.per_step_reward
        if p is None:
            raise ValueError("per_step_reward is None; config not fully initialized")
        self._per_step_reward = p

    def reward(self, state: str, action: str, step_idx: int) -> tuple[float, bool]:
        return self._per_step_reward[step_idx][state], False


def register() -> None:
    from rl_health_interventions.rewards import REGISTRY

    REGISTRY["compound"] = CompoundReward
