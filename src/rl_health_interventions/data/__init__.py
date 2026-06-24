from __future__ import annotations

import logging
from typing import Type

from rl_health_interventions.data import synthetic, feature_pipeline, nhanes, weather

logger = logging.getLogger(__name__)

REGISTRY: dict[str, Type] = {}


def make(name: str, **kwargs) -> object:
    if name not in REGISTRY:
        raise KeyError(f"Unknown data component: {name}. Known: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


_DATA_COMPONENT_MODULES = [synthetic, feature_pipeline, nhanes, weather]

for _mod in _DATA_COMPONENT_MODULES:
    try:
        _mod.register()
    except Exception:
        logger.exception("Failed to register %s", _mod.__name__)
