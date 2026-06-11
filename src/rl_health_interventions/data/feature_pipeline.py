from __future__ import annotations

from typing import Any


class FeaturePipeline:
    def __init__(self) -> None:
        self._transforms: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> FeaturePipeline:
        # Stub: config is accepted but ignored. Real implementation in subphase 1A.
        return cls()


def register() -> None:
    from rl_health_interventions.data import REGISTRY

    REGISTRY[FeaturePipeline.__name__] = FeaturePipeline
