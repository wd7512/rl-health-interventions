from __future__ import annotations

import logging
from typing import Type

from rl_health_interventions.data import synthetic, feature_pipeline

logger = logging.getLogger(__name__)

REGISTRY: dict[str, Type] = {}


def make(name: str) -> object:
    if name not in REGISTRY:
        raise KeyError(f"Unknown data component: {name}. Known: {list(REGISTRY)}")
    return REGISTRY[name]()


try:
    synthetic.register()
except Exception:
    logger.exception("Failed to register data.synthetic")

try:
    feature_pipeline.register()
except Exception:
    logger.exception("Failed to register data.feature_pipeline")
