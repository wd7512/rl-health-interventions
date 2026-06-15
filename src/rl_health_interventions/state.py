from __future__ import annotations
from dataclasses import dataclass
from rl_health_interventions.config.schemas import ActivityLevel, TimeOfDay


@dataclass(frozen=True)
class StateView:
    activity: ActivityLevel
    time_of_day: TimeOfDay
    day: int
    step_of_day: int

    @property
    def global_step(self) -> int:
        return self.day * 5 + self.step_of_day
