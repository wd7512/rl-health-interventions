from __future__ import annotations

from rl_health_interventions.config.schemas import ActivityLevel, Action
from rl_health_interventions.rewards._base import RewardHandler


class CompoundReward(RewardHandler):
    def reward(
        self, state: ActivityLevel, action: Action
    ) -> tuple[float, bool]:
        return 0.0, False


def register() -> None:
    from rl_health_interventions.rewards import REGISTRY

    REGISTRY["compound"] = CompoundReward
