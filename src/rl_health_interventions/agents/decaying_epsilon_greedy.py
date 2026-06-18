from __future__ import annotations

from rl_health_interventions.agents.contextual_bandits.decaying_epsilon_greedy import (  # noqa: F401
    DecayingEpsilonGreedyAgent,
)


def register() -> None:
    """
    Register the DecayingEpsilonGreedyAgent into the agent registry.
    """
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["decaying_epsilon_greedy"] = DecayingEpsilonGreedyAgent
