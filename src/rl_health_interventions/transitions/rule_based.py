from __future__ import annotations

from rl_health_interventions.transitions._base import TransitionModel


class RuleBasedTransition(TransitionModel):
    def transition(self, state: object, action: int, profile: object) -> object:
        return state


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY[RuleBasedTransition.__name__] = RuleBasedTransition
