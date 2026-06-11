from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ResponseModel(ABC):
    @abstractmethod
    def response(self, state: Any, action: int, profile: Any) -> float: ...
