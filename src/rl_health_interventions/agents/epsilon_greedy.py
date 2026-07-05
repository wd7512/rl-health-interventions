from __future__ import annotations

from rl_health_interventions.agents.contextual_bandits.epsilon_greedy import (
    EpsilonGreedyAgent,
)


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["epsilon_greedy"] = EpsilonGreedyAgent
