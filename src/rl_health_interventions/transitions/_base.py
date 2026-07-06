from __future__ import annotations

from abc import ABC, abstractmethod

from rl_health_interventions.state import StateView


class TransitionModel(ABC):
    @abstractmethod
    def transition(self, state: StateView, action: str) -> dict[str, str]: ...

    def _sample_sleep(self, state: StateView) -> str:  # noqa: ARG002
        msg = f"{type(self).__name__} does not support sleep transitions"
        raise RuntimeError(msg)

    def _sample_step_bin(self, state: StateView, action: str, k: int) -> str:  # noqa: ARG002
        msg = f"{type(self).__name__} does not support step_bin transitions"
        raise RuntimeError(msg)
