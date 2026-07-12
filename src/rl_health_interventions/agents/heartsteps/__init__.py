from __future__ import annotations

from rl_health_interventions.agents.heartsteps.agent import HeartStepsAgent


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY.register("heartsteps", HeartStepsAgent)
