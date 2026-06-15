from __future__ import annotations

from abc import ABC, abstractmethod

from rl_health_interventions.config.schemas import ActivityLevel, Action, TimeOfDay


class TransitionModel(ABC):
    @abstractmethod
    def transition(
        self, state: ActivityLevel, action: Action, time_of_day: TimeOfDay
    ) -> ActivityLevel: ...
