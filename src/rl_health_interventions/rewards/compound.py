from __future__ import annotations

from rl_health_interventions.config.schemas import ActivityLevel, Action, MDPConfig
from rl_health_interventions.rewards._base import RewardHandler


class CompoundReward(RewardHandler):
    def __init__(self, config: MDPConfig | None = None) -> None:
        self._config = config

    def reward(self, state: ActivityLevel, action: Action) -> tuple[float, bool]:
        if self._config is None:
            return 0.0, False
        if state == ActivityLevel.ACTIVE:
            return self._config.reward_active, False
        return self._config.reward_sedentary, False


def register() -> None:
    from rl_health_interventions.rewards import REGISTRY

    REGISTRY["compound"] = CompoundReward
