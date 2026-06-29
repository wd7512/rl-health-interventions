from __future__ import annotations

from abc import ABC, abstractmethod

from rl_health_interventions.state import StateView


class RewardHandler(ABC):
    @abstractmethod
    def reward(
        self, state: StateView, action: str, step_idx: int
    ) -> tuple[float, bool]: ...
