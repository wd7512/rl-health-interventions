from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RewardHandler(ABC):
    @abstractmethod
    def reward(self, state: Any, action: int, profile: Any) -> tuple[float, bool]: ...
