from __future__ import annotations

from rl_health_interventions.agents._base import Agent


class FixedAgent(Agent):
    def __init__(
        self,
        action: str = "idle",
        seed: int | None = None,  # Unused — satisfies Agent interface contract
        actions: list[str] | None = None,  # Unused — satisfies Agent interface contract
    ) -> None:
        self._action = action

    def select_action(self, state) -> str:
        return self._action


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY.register("fixed", FixedAgent)
