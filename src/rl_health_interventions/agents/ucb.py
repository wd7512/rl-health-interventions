from __future__ import annotations

from rl_health_interventions.agents.contextual_bandits.ucb import (  # noqa: F401
    UCBAgent,
)


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["ucb"] = UCBAgent
