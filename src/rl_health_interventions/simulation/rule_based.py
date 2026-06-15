from __future__ import annotations

from rl_health_interventions.config.schemas import ActivityLevel, Action
from rl_health_interventions.simulation._base import ResponseModel


class RuleBasedResponse(ResponseModel):
    def response(self, state: ActivityLevel, action: Action) -> float:
        return 0.0


def register() -> None:
    from rl_health_interventions.simulation import REGISTRY

    REGISTRY["rule_based"] = RuleBasedResponse
