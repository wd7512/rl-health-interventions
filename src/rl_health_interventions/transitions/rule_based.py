from __future__ import annotations

import numpy as np
from typing_extensions import override

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions._base import TransitionModel


class RuleBasedTransition(TransitionModel):
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        super().__init__(config, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._cache: dict[tuple[str, str], tuple[list[str], np.ndarray]] = {}
        self._build_cache(config)

    def _build_cache(self, config: MDPConfig) -> None:
        prob_config = config.transition_model.transition_probabilities
        if prob_config is None:
            raise ValueError("rule_based transition requires transition_probabilities")
        for state, actions in prob_config.root.items():
            for action, probs in actions.items():
                targets = list(probs.keys())
                prob_values = np.array(list(probs.values()), dtype=np.float64)
                prob_values /= prob_values.sum()
                self._cache[(state, action)] = (targets, prob_values)

    @override
    def transition(self, state: StateView, action: str) -> dict[str, str]:
        updates: dict[str, str] = {}
        for factor_name in self._stochastic_factors:
            current_val = getattr(state, factor_name)
            targets, probs = self._cache[(current_val, action)]
            idx = self._rng.choice(len(targets), p=probs)
            updates[factor_name] = targets[idx]
        return updates


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY.register("rule_based", RuleBasedTransition)
