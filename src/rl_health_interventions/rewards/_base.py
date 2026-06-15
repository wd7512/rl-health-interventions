from __future__ import annotations

from abc import ABC, abstractmethod

from rl_health_interventions.config.schemas import ActivityLevel, Action


class RewardHandler(ABC):
    @abstractmethod
    def reward(self, state: ActivityLevel, action: Action) -> tuple[float, bool]: ...
