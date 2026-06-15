from __future__ import annotations

from typing import Any

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.config.schemas import Action


class ThompsonSamplingAgent(Agent):
    def select_action(self, state: Any) -> Action:
        return Action.SEND

    def update(self, state: Any, action: Action, reward: float, next_state: Any) -> None:
        pass


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["thompson_sampling"] = ThompsonSamplingAgent
