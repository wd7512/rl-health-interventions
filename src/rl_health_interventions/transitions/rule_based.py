from __future__ import annotations

from typing import Any

from rl_health_interventions.transitions._base import TransitionModel


class RuleBasedTransition(TransitionModel):
    def transition(self, state: Any, action: int, profile: Any) -> Any:
        return state


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY[RuleBasedTransition.__name__] = RuleBasedTransition
