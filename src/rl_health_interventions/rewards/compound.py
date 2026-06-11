from __future__ import annotations

from typing import Any

from rl_health_interventions.rewards._base import RewardHandler


class CompoundReward(RewardHandler):
    def reward(self, state: Any, action: int, profile: Any) -> tuple[float, bool]:
        return 0.0, False


def register() -> None:
    from rl_health_interventions.rewards import REGISTRY

    REGISTRY["compound"] = CompoundReward
