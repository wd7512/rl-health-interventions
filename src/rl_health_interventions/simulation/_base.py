from __future__ import annotations

from abc import ABC, abstractmethod

from rl_health_interventions.config.schemas import ActivityLevel, Action


class ResponseModel(ABC):
    @abstractmethod
    def response(
        self, state: ActivityLevel, action: Action
    ) -> float: ...
