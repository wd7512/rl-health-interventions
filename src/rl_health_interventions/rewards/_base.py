from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RewardHandler(ABC):
    @abstractmethod
    def reward(self, state: Any, action: str, step_idx: int) -> tuple[float, bool]: ...
