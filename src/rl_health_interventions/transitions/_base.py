from __future__ import annotations

from abc import ABC, abstractmethod

from rl_health_interventions.state import StateView


class TransitionModel(ABC):
    @abstractmethod
    def transition(self, state: StateView, action: str) -> StateView: ...
