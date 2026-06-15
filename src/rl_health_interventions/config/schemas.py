from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, RootModel, model_validator


class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"
    ACTIVE = "active"


class TimeOfDay(str, Enum):
    MORNING = "morning"
    MIDDAY = "midday"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


class Action(str, Enum):
    SEND = "send"
    DON_T_SEND = "don_t_send"


class TransitionMatrix(RootModel):
    root: dict[ActivityLevel, dict[Action, dict[ActivityLevel, float]]]

    @model_validator(mode="after")
    def _check_probabilities_sum_to_one(self):
        for state, actions in self.root.items():
            for action, probs in actions.items():
                total = sum(probs.values())
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(
                        f"Probabilities for ({state}, {action}) sum to {total}, expected 1.0"
                    )
        return self


class TimeOfDayMask(RootModel):
    root: dict[TimeOfDay, dict[ActivityLevel, float]]


class MDPConfig(BaseModel):
    activity_levels: list[ActivityLevel]
    actions: list[Action]
    time_of_day: list[TimeOfDay]
    steps_per_day: int = Field(ge=1, le=24, default=5)
    episode_days: int = Field(ge=1, le=365, default=90)
    transition: TransitionMatrix
    masks: TimeOfDayMask
    initial_state: ActivityLevel = ActivityLevel.SEDENTARY
    reward_active: float = 1.0
    reward_sedentary: float = 0.0
    seed: int = 42
