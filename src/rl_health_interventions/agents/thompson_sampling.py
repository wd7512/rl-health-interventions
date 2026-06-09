from __future__ import annotations

from typing import Any

from rl_health_interventions.agents._base import Agent


class ThompsonSamplingAgent(Agent):
    def select_action(self, state: Any) -> int:
        return 0

    def update(self, state: Any, action: int, reward: float, next_state: Any) -> None:
        pass


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY[ThompsonSamplingAgent.__name__] = ThompsonSamplingAgent
