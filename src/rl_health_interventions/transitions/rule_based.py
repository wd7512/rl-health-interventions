from __future__ import annotations

import numpy as np

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.transitions._base import TransitionModel


class RuleBasedTransition(TransitionModel):
    def __init__(self, config: MDPConfig, seed: int = 42, **_kwargs: object) -> None:
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

    def transition(self, state: str, action: str) -> str:
        targets, probs = self._cache[(state, action)]
        idx = self._rng.choice(len(targets), p=probs)
        return targets[idx]


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY["rule_based"] = RuleBasedTransition
