from __future__ import annotations

from typing_extensions import override

from rl_health_interventions.simulation._base import ResponseModel


class RuleBasedResponse(ResponseModel):
    @override
    def response(self, state: str, action: str) -> float:
        return 0.0


def register() -> None:
    from rl_health_interventions.simulation import REGISTRY

    REGISTRY["rule_based"] = RuleBasedResponse
