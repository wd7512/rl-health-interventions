from __future__ import annotations

import logging
from typing import Type

from rl_health_interventions.transitions._base import TransitionModel
from rl_health_interventions.transitions import rule_based

logger = logging.getLogger(__name__)

REGISTRY: dict[str, Type[TransitionModel]] = {}


def register(cls: Type[TransitionModel]) -> None:
    REGISTRY[cls.__name__] = cls


def make(name: str) -> TransitionModel:
    if name not in REGISTRY:
        raise KeyError(f"Unknown transition model: {name}. Known: {list(REGISTRY)}")
    return REGISTRY[name]()


try:
    rule_based.register()
except Exception:
    logger.exception("Failed to register rule_based transition model")
