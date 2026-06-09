from __future__ import annotations

from rl_health_interventions.rewards._base import RewardHandler


class CompoundReward(RewardHandler):
    def reward(self, state: object, action: int, profile: object) -> tuple[float, bool]:
        return 0.0, False


def register() -> None:
    from rl_health_interventions.rewards import REGISTRY

    REGISTRY[CompoundReward.__name__] = CompoundReward
