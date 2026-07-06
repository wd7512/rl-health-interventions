from __future__ import annotations

from typing_extensions import override

from rl_health_interventions.agents._base import Agent


class FixedAgent(Agent):
    def __init__(
        self,
        action: str = "idle",
        seed: int | None = None,  # noqa: ARG002
        actions: list[str] | None = None,  # noqa: ARG002
    ) -> None:
        self._action = action

    @override
    def select_action(self, state) -> str:
        return self._action


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY.register("fixed", FixedAgent)
