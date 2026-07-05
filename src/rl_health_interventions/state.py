from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StateView:
    activity: str
    day: int
    step_of_day: int
    steps_per_day: int = field(default=5)

    @property
    def global_step(self) -> int:
        return self.day * self.steps_per_day + self.step_of_day
