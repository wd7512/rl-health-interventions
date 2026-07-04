from __future__ import annotations

from rl_health_interventions.agents.contextual_bandits.epsilon_greedy import (
    EpsilonGreedyAgent,
)
from rl_health_interventions.agents.contextual_bandits.ucb import UCBAgent
from rl_health_interventions.agents.contextual_bandits.thompson_sampling import (
    ThompsonSamplingAgent,
)
from rl_health_interventions.agents.contextual_bandits.decaying_epsilon_greedy import (
    DecayingEpsilonGreedyAgent,
)


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY.register("epsilon_greedy", EpsilonGreedyAgent)
    REGISTRY.register("ucb", UCBAgent)
    REGISTRY.register("thompson_sampling", ThompsonSamplingAgent)
    REGISTRY.register("decaying_epsilon_greedy", DecayingEpsilonGreedyAgent)
