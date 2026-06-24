from __future__ import annotations

from abc import ABC, abstractmethod


class RewardHandler(ABC):
    @abstractmethod
    def reward(self, state: str, action: str, step_idx: int) -> tuple[float, bool]: ...
