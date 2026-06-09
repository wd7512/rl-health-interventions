from __future__ import annotations

import logging
from typing import Type

from rl_health_interventions.data._base import DatasetProtocol
from rl_health_interventions.data import synthetic, feature_pipeline

logger = logging.getLogger(__name__)

REGISTRY: dict[str, Type[DatasetProtocol]] = {}


def make(name: str) -> DatasetProtocol:
    if name not in REGISTRY:
        raise KeyError(f"Unknown data component: {name}. Known: {list(REGISTRY)}")
    return REGISTRY[name]()


try:
    synthetic.register()
except Exception:
    logger.exception("Failed to register SyntheticDataGenerator")

try:
    feature_pipeline.register()
except Exception:
    logger.exception("Failed to register FeaturePipeline")
