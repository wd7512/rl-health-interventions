from __future__ import annotations

from rl_health_interventions.agents._base import Agent


class ThompsonSamplingAgent(Agent):
    def select_action(self, state: object) -> int:
        return 0

    def update(
        self, state: object, action: int, reward: float, next_state: object
    ) -> None:
        pass


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY[ThompsonSamplingAgent.__name__] = ThompsonSamplingAgent
