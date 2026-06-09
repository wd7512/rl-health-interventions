from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TransitionModel(ABC):
    @abstractmethod
    def transition(self, state: Any, action: int, profile: Any) -> Any: ...
