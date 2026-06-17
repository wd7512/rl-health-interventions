from __future__ import annotations

from abc import ABC, abstractmethod


class ResponseModel(ABC):
    @abstractmethod
    def response(self, state: str, action: str) -> float: ...
