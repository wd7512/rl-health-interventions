from __future__ import annotations

from rl_health_interventions.simulation._base import ResponseModel


class RuleBasedResponse(ResponseModel):
    def response(self, state: object, action: int, profile: object) -> object:
        return 0.0


def register() -> None:
    from rl_health_interventions.simulation import REGISTRY

    REGISTRY[RuleBasedResponse.__name__] = RuleBasedResponse
