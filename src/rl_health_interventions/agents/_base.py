from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rl_health_interventions.config.schemas import Action


class Agent(ABC):
    @abstractmethod
    def select_action(self, state: Any) -> Action: ...

    def update(
        self, state: Any, action: Action, reward: float, next_state: Any
    ) -> None: ...
