from __future__ import annotations

from rl_health_interventions.agents.contextual_bandits.ucb import (
    UCBAgent,
)


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["ucb"] = UCBAgent
