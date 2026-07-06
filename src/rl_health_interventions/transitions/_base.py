from __future__ import annotations

from abc import ABC, abstractmethod

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView


class TransitionModel(ABC):
    @abstractmethod
    def transition(self, state: StateView, action: str) -> dict[str, str]: ...

    @property
    def _stochastic_factors(self) -> list[str]:
        return [
            n for n, c in self._config.state.variables.items() if c.advanced is None
        ]

    def __init__(self, config: MDPConfig, seed: int = 42) -> None:  # noqa: ARG002
        self._config = config
