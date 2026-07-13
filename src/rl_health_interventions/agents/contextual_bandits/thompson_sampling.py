from __future__ import annotations

from typing import NamedTuple

import numpy as np

from typing_extensions import override

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
    _NumpyDictWrapper,
)


class Posterior(NamedTuple):
    alpha: float
    beta: float


class _PosteriorDictWrapper:
    """Wrapper to provide dict-like access to NumPy arrays for Thompson Sampling posteriors."""

    def __init__(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        action_to_index: dict[str, int],
    ) -> None:
        self.alphas = alphas
        self.betas = betas
        self.action_to_index = action_to_index

    def __getitem__(self, key: str) -> Posterior:
        if isinstance(key, str):
            idx = self.action_to_index[key]
            return Posterior(alpha=self.alphas[idx], beta=self.betas[idx])
        raise KeyError(key)

    def __setitem__(self, key: str, value: Posterior) -> None:
        if isinstance(key, str):
            idx = self.action_to_index[key]
            self.alphas[idx] = value.alpha
            self.betas[idx] = value.beta
        else:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return isinstance(key, str) and key in self.action_to_index

    def __repr__(self) -> str:
        return repr({a: Posterior(alpha=self.alphas[i], beta=self.betas[i])
                     for a, i in self.action_to_index.items()})

    def get(self, key: str, default: object = None) -> object:
        if isinstance(key, str) and key in self.action_to_index:
            idx = self.action_to_index[key]
            return Posterior(alpha=self.alphas[idx], beta=self.betas[idx])
        return default

    def keys(self) -> list[str]:
        return list(self.action_to_index.keys())

    def values(self) -> list[Posterior]:
        return [Posterior(alpha=self.alphas[self.action_to_index[a]], beta=self.betas[self.action_to_index[a]])
                for a in self.action_to_index]

    def items(self) -> list[tuple[str, Posterior]]:
        return [(a, Posterior(alpha=self.alphas[self.action_to_index[a]], beta=self.betas[self.action_to_index[a]]))
                for a in self.action_to_index]


class ThompsonSamplingAgent(ContextualBanditAgent):
    """Beta-Bernoulli Thompson Sampling for binary actions.

    When ``contextual=True``, maintains separate ``(alpha, beta)``
    posteriors for each ``(context_value, action)`` pair rather than
    per-action globally.

    For non-contextual mode, uses NumPy arrays internally for alphas and betas
    for better performance, but provides dict-like access for backward compatibility.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int = 42,
        contextual: bool = False,
        context_features: str | list[str] | None = None,
    ) -> None:
        super().__init__(
            actions=actions,
            seed=seed,
            contextual=contextual,
            context_features=context_features,
        )
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError("alpha_prior and beta_prior must be strictly positive.")
        self._init_params()

    def _init_params(self) -> None:
        if self._use_numpy:
            # Use NumPy arrays for non-contextual mode
            self.alphas_array: np.ndarray = np.full(
                self._n_actions, self.alpha_prior, dtype=np.float64
            )
            self.betas_array: np.ndarray = np.full(
                self._n_actions, self.beta_prior, dtype=np.float64
            )
            # Provide dict-like access for backward compatibility
            self.posteriors: _PosteriorDictWrapper = _PosteriorDictWrapper(
                self.alphas_array, self.betas_array, self._action_to_index
            )
        else:
            # Use dicts for contextual mode
            self.posteriors: dict = {}

    def _ensure_params(self, key: str | tuple[str, ...]) -> None:
        if self._use_numpy:
            return  # NumPy arrays are pre-allocated
        if key not in self.posteriors:
            self.posteriors[key] = Posterior(
                alpha=self.alpha_prior, beta=self.beta_prior
            )

    @override
    def select_action(self, state) -> str:
        if self._use_numpy:
            # Vectorized sampling for non-contextual mode
            samples = self._rng.beta(self.alphas_array, self.betas_array)
            best_idx = int(np.argmax(samples))
            return self._actions[best_idx]
        else:
            # Dict-based for contextual mode
            samples = {}
            for action in self._actions:
                key = self._get_context_key(state, action)
                self._ensure_params(key)
                posterior = self.posteriors[key]
                samples[action] = float(self._rng.beta(posterior.alpha, posterior.beta))
            return max(samples, key=lambda a: samples[a])

    @override
    def update(self, state, action: str, reward: float, next_state) -> None:
        if self._use_numpy:
            # For non-contextual, use action index directly
            action_idx = self._action_to_index[action]
            if reward > 0.0:
                self.alphas_array[action_idx] += 1
            else:
                self.betas_array[action_idx] += 1
        else:
            # For contextual, use context key
            key = self._get_context_key(state, action)
            self._ensure_params(key)
            p = self.posteriors[key]
            if reward > 0.0:
                self.posteriors[key] = Posterior(alpha=p.alpha + 1, beta=p.beta)
            else:
                self.posteriors[key] = Posterior(alpha=p.alpha, beta=p.beta + 1)
