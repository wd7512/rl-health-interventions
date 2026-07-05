from __future__ import annotations

import logging
from typing import Type

from rl_health_interventions.rewards import compound
from rl_health_interventions.rewards._base import RewardHandler

logger = logging.getLogger(__name__)

REGISTRY: dict[str, type[RewardHandler]] = {}


def make(name_or_config=None, **kwargs) -> RewardHandler:
    if name_or_config is not None and not isinstance(name_or_config, str):
        name = "compound"
        kwargs.setdefault("config", name_or_config)
    elif isinstance(name_or_config, str):
        name = name_or_config
    elif "name" in kwargs:
        name = kwargs.pop("name")
    else:
        raise TypeError("make() requires either a config or name argument")
    if name not in REGISTRY:
        raise KeyError(f"Unknown reward handler: {name}. Known: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


# NOTE: Import new reward handler module above and append it here so register() runs on import.
# Each module must have a register() function that adds to REGISTRY.
_REWARD_MODULES = [compound]

for _mod in _REWARD_MODULES:
    try:
        _mod.register()
    except Exception:
        logger.exception("Failed to register %s", _mod.__name__)
