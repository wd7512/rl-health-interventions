from __future__ import annotations

from abc import ABC, abstractmethod

from rl_health_interventions.state import StateView


class TransitionModel(ABC):
    @abstractmethod
    def _transition_updates(self, state: StateView, action: str) -> dict[str, str]: ...

    def transition(self, state: StateView, action: str) -> StateView:
        return state.with_factors(**self._transition_updates(state, action))
