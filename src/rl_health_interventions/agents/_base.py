from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    @abstractmethod
    def select_action(self, state: Any) -> int: ...

    def update(
        self, state: Any, action: int, reward: float, next_state: Any
    ) -> None: ...
