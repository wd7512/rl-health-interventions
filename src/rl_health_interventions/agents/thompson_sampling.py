from __future__ import annotations

from rl_health_interventions.agents.contextual_bandits.thompson_sampling import (  # noqa: F401
    Posterior,
    ThompsonSamplingAgent,
)


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["thompson_sampling"] = ThompsonSamplingAgent
