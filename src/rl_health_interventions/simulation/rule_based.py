from __future__ import annotations

from typing import Any

from rl_health_interventions.simulation._base import ResponseModel


class RuleBasedResponse(ResponseModel):
    def response(self, state: Any, action: int, profile: Any) -> float:
        return 0.0


def register() -> None:
    from rl_health_interventions.simulation import REGISTRY

    REGISTRY["rule_based"] = RuleBasedResponse
