from __future__ import annotations

import logging
from typing import Type

from rl_health_interventions.simulation._base import ResponseModel
from rl_health_interventions.simulation import rule_based

logger = logging.getLogger(__name__)

REGISTRY: dict[str, Type[ResponseModel]] = {}


def make(name: str, **kwargs) -> ResponseModel:
    if name not in REGISTRY:
        raise KeyError(f"Unknown response model: {name}. Known: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


try:
    rule_based.register()
except Exception:
    logger.exception("Failed to register rule_based response model")
