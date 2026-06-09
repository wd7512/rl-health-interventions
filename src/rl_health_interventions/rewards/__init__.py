from __future__ import annotations

import logging
from typing import Type

from rl_health_interventions.rewards._base import RewardHandler
from rl_health_interventions.rewards import compound

logger = logging.getLogger(__name__)

REGISTRY: dict[str, Type[RewardHandler]] = {}


def register(cls: Type[RewardHandler]) -> None:
    REGISTRY[cls.__name__] = cls


def make(name: str) -> RewardHandler:
    if name not in REGISTRY:
        raise KeyError(f"Unknown reward handler: {name}. Known: {list(REGISTRY)}")
    return REGISTRY[name]()


try:
    compound.register()
except Exception:
    logger.exception("Failed to register compound reward handler")
