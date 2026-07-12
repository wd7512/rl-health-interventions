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
        sigma_sq: float = 1.0,
        reference_action: str | None = None,
        prior_mean: float = 0.0,
        prior_cov: float = 1.0,
        **_kwargs: object,
    ) -> None:
        self._actions = actions or ["idle", "nudge"]
        self._rng = np.random.default_rng(seed)
        self._reference_action = reference_action or self._actions[0]
        self._feature_names: tuple[str, ...] = ()
        self._n_features = 0
        self._feature_index: dict[str, int] = {}
        self._one_hot_map: dict[str, dict[str | int, int]] = {}
        self._total_one_hot_dim = 0
        self._dosage_tracker = DosageTracker(lambda_decay=lambda_dosage)
        self._sigma_sq = sigma_sq
        self._prior_mean = prior_mean
        self._prior_cov = prior_cov
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
        self._feature_sum = np.zeros(0, dtype=np.float64)
        self._feature_count = 0

    def init_one_hot_map(
        self,
        state_variables: dict[str, list[str]],
        extra_features: dict[str, list[int]] | None = None,
    ) -> None:
        """Build one-hot encoding map from state variable definitions.

        Args:
            state_variables: MDP state variables (str-valued).
            extra_features: Additional integer-valued features (e.g. step_of_day).
        """
        self._one_hot_map = {}
        self._feature_names_list: list[str] = []
        offset = 0
        for var_name, var_values in state_variables.items():
            self._one_hot_map[var_name] = {
                val: offset + i for i, val in enumerate(var_values)
            }
            offset += len(var_values)
            self._feature_names_list.extend(f"{var_name}_{val}" for val in var_values)
        for feat_name, feat_values in (extra_features or {}).items():
            self._one_hot_map[feat_name] = {
                val: offset + i for i, val in enumerate(feat_values)
            }
            offset += len(feat_values)
            self._feature_names_list.extend(f"{feat_name}_{val}" for val in feat_values)
        self._total_one_hot_dim = offset
        self._n_features = self._total_one_hot_dim
        self._feature_names = tuple(self._feature_names_list)
        self._feature_index = {name: i for i, name in enumerate(self._feature_names)}
        self._regression = MultiClassBayesianRegression(
            n_features=self._n_features,
            actions=self._actions,
            reference_action=self._reference_action,
            sigma_sq=self._sigma_sq,
            prior_mean=self._prior_mean,
            prior_cov=self._prior_cov,
        )
        self._feature_sum = np.zeros(self._n_features, dtype=np.float64)
        self._feature_count = 0

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
        q_means = self._regression.get_reward_means(avg_features=f)
        for a in self._actions:
            q_means[a] += self._proxy.get_eta(a, dosage)
        return max(q_means, key=lambda a: q_means[a])

    @override
    def update(self, state, action: str, reward: float, next_state) -> None:
        assert self._regression is not None, "Call init_one_hot_map first"
        f = self._one_hot(state)
        suggestion = action != self._reference_action
        self._dosage_tracker.update(suggestion)
        self._buffer.append((f, action, reward))
        self._feature_sum += f
        self._feature_count += 1

    @override
    def on_day_end(self) -> None:
        if not self._buffer:
            return
        assert self._regression is not None
        self._regression.update_batch(self._buffer)
        avg_features = (
            self._feature_sum / self._feature_count if self._feature_count > 0 else None
        )
        reward_means = self._regression.get_reward_means(avg_features)
        self._proxy.update(reward_means)
        self._buffer.clear()
        self._feature_sum = np.zeros(self._n_features, dtype=np.float64)
        self._feature_count = 0

    def get_beta_means(self) -> dict[str, np.ndarray]:
        """Return posterior means for each action."""
        assert self._regression is not None, "Call init_one_hot_map first"
        return self._regression.get_beta_means()

    def get_proxy_table(self) -> np.ndarray:
        """Return the proxy value table H."""
        return self._proxy.value_table

    def get_dosage(self) -> float:
        """Return current dosage value."""
        return self._dosage_tracker.get_dosage()

    def get_gamma(self) -> float:
        """Return the discount factor."""
        return self._gamma

    def get_action_index(self) -> dict[str, int]:
        """Return mapping from action name to index."""
        return self._proxy._action_idx
