from __future__ import annotations

import numpy as np
from typing_extensions import override

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions._base import TransitionModel


class RandomTransition(TransitionModel):
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        super().__init__(config, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self._build_cache()

    def _build_cache(self) -> None:
        """Build Dirichlet-randomised probability tables for all stochastic factors."""
        for name in self._stochastic_factors:
            var_cfg = self._config.state.variables.get(name)
            if var_cfg is None:
                continue
            n = len(var_cfg.names)
            self._cache[name] = (
                var_cfg.names,
                self._rng.dirichlet([1.0] * n),
            )

    def _sample(self, factor_name: str, state: StateView, action: str) -> str:  # noqa: ARG002
        targets, probs = self._cache[factor_name]
        idx = self._rng.choice(len(targets), p=probs)
        return str(targets[idx])

    @override
    def transition(self, state: StateView, action: str) -> dict[str, str]:
        updates: dict[str, str] = {}
        if state.step_of_day == 0 and "sleep" in self._cache:
            updates["sleep"] = self._sample("sleep", state, action)
            state = state.with_factors(sleep=updates["sleep"])
        for name in self._stochastic_factors:
            if name == "sleep":
                continue
            updates[name] = self._sample(name, state, action)
        return updates


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY.register("random", RandomTransition)
