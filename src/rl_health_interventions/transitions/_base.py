from __future__ import annotations

from abc import ABC, abstractmethod


class TransitionModel(ABC):
    @abstractmethod
    def transition(self, state: str, action: str) -> str: ...
