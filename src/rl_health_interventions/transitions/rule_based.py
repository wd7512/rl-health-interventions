from __future__ import annotations

from rl_health_interventions.config.schemas import ActivityLevel, Action, TimeOfDay
from rl_health_interventions.transitions._base import TransitionModel


class RuleBasedTransition(TransitionModel):
    def transition(
        self, state: ActivityLevel, action: Action, time_of_day: TimeOfDay
    ) -> ActivityLevel:
        return state


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY["rule_based"] = RuleBasedTransition
