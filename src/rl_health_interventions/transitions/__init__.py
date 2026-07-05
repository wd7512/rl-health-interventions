from __future__ import annotations

import logging

from rl_health_interventions.transitions import rule_based
from rl_health_interventions.transitions._base import TransitionModel

logger = logging.getLogger(__name__)

REGISTRY: dict[str, type[TransitionModel]] = {}


def make(name_or_config=None, **kwargs) -> TransitionModel:
    if name_or_config is not None and not isinstance(name_or_config, str):
        name = name_or_config.transition_model.type
        kwargs.setdefault("config", name_or_config)
    elif isinstance(name_or_config, str):
        name = name_or_config
    elif "name" in kwargs:
        name = kwargs.pop("name")
    else:
        raise TypeError("make() requires either a config or name argument")
    if name not in REGISTRY:
        _msg = f"Unknown transition model: {name}. Known: {list(REGISTRY)}"
        raise KeyError(_msg)
    return REGISTRY[name](**kwargs)


# NOTE: Import new transition model module above and append it here
# so register() runs on import. Each module must have a register()
# function that adds to REGISTRY.
_TRANSITION_MODULES = [rule_based]

for _mod in _TRANSITION_MODULES:
    try:
        _mod.register()
    except Exception:
        logger.exception("Failed to register %s", _mod.__name__)
