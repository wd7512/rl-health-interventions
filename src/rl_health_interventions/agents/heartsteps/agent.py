from __future__ import annotations

import numpy as np
from typing_extensions import override

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.agents.heartsteps.bayesian_regression import (
    MultiClassBayesianRegression,
)
from rl_health_interventions.agents.heartsteps.dosage import DosageTracker
from rl_health_interventions.agents.heartsteps.proxy_value import ProxyValue


class HeartStepsAgent(Agent):
    """HeartSteps V2 RL agent (Liao et al. 2019).

    Bayesian linear regression with action-centering, dosage tracking,
    and a proxy value function for delayed effects.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        seed: int = 42,
        gamma: float = 0.99,
        lambda_dosage: float = 0.95,
        w: float = 0.5,
        epsilon_0: float = 0.2,
        epsilon_1: float = 0.1,
        sigma_sq: float = 1.0,
        reference_action: str | None = None,
        _prior_mean: float = 0.0,
        _prior_cov: float = 1.0,
        _f_features: str | list[str] | None = None,
        _g_features: str | list[str] | None = None,
        **_kwargs: object,
    ) -> None:
        self._actions = actions or ["idle", "nudge"]
        self._rng = np.random.default_rng(seed)
        self._reference_action = reference_action or self._actions[0]
        self._feature_names: tuple[str, ...] = ()
        self._n_features = 0
        self._feature_index: dict[str, int] = {}
        self._one_hot_map: dict[str, dict[str, int]] = {}
        self._total_one_hot_dim = 0
        self._dosage_tracker = DosageTracker(lambda_decay=lambda_dosage)
        self._sigma_sq = sigma_sq
        self._regression: MultiClassBayesianRegression | None = None
        self._proxy = ProxyValue(
            actions=self._actions,
            reference_action=self._reference_action,
            gamma=gamma,
            w=w,
            lambda_dosage=lambda_dosage,
        )
        self._buffer: list[tuple] = []
        self._gamma = gamma
        self._epsilon_0 = epsilon_0
        self._epsilon_1 = epsilon_1

    def init_one_hot_map(self, state_variables: dict[str, list[str]]) -> None:
        """Build one-hot encoding map from state variable definitions."""
        self._one_hot_map = {}
        offset = 0
        for var_name, var_values in state_variables.items():
            self._one_hot_map[var_name] = {
                val: offset + i for i, val in enumerate(var_values)
            }
            offset += len(var_values)
        self._total_one_hot_dim = offset
        self._n_features = self._total_one_hot_dim
        self._feature_names = tuple(
            f"{var}_{val}" for var, vals in state_variables.items() for val in vals
        )
        self._feature_index = {name: i for i, name in enumerate(self._feature_names)}
        self._regression = MultiClassBayesianRegression(
            n_features=self._n_features,
            actions=self._actions,
            reference_action=self._reference_action,
            sigma_sq=self._sigma_sq,
        )

    def _one_hot(self, state) -> np.ndarray:
        """Encode state as a one-hot feature vector."""
        vec = np.zeros(self._total_one_hot_dim, dtype=np.float64)
        for var_name, val_map in self._one_hot_map.items():
            state_val = getattr(state, var_name, None)
            if state_val is not None and state_val in val_map:
                vec[val_map[state_val]] = 1.0
        return vec

    @override
    def select_action(self, state) -> str:
        assert self._regression is not None, "Call init_one_hot_map first"
        f = self._one_hot(state)
        dosage = self._dosage_tracker.get_dosage()
        samples = self._regression.sample_betas(self._rng)

        q_values = {}
        for action in self._actions:
            eta = self._proxy.get_eta(action, dosage)
            q_values[action] = float(f @ samples[action]) + self._gamma * eta

        actions_list = list(q_values.keys())
        q_arr = np.array([q_values[a] for a in actions_list])
        softmax = np.exp(q_arr - np.max(q_arr))
        softmax = softmax / softmax.sum()
        clipped = np.clip(softmax, self._epsilon_1, 1.0 - self._epsilon_0)
        clipped = clipped / clipped.sum()
        chosen_idx = int(self._rng.choice(len(actions_list), p=clipped))
        return actions_list[chosen_idx]

    @override
    def update(self, state, action: str, reward: float, next_state) -> None:
        assert self._regression is not None, "Call init_one_hot_map first"
        f = self._one_hot(state)
        suggestion = action != self._reference_action
        self._dosage_tracker.update(suggestion)
        self._buffer.append((f, action, reward))

    @override
    def on_day_end(self) -> None:
        if not self._buffer:
            return
        assert self._regression is not None
        self._regression.update_batch(self._buffer)
        reward_means = self._regression.get_reward_means()
        self._proxy.update(reward_means)
        self._buffer.clear()
