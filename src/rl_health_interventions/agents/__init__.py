from __future__ import annotations

import logging
from typing import Type

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.agents import epsilon_greedy
from rl_health_interventions.agents import thompson_sampling

logger = logging.getLogger(__name__)

REGISTRY: dict[str, Type[Agent]] = {}


def make(name: str, **kwargs) -> Agent:
    if name not in REGISTRY:
        raise KeyError(f"Unknown agent: {name}. Known: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


try:
    thompson_sampling.register()
except Exception:
    logger.exception("Failed to register thompson_sampling agent")

try:
    epsilon_greedy.register()
except Exception:
    logger.exception("Failed to register epsilon_greedy agent")
