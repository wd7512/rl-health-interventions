"""Contextual bandit agent using NormWear FM embeddings as context.

This is the real FM integration — generates synthetic sensor signals
from state, feeds them through NormWear, and uses the resulting 768-dim
embedding (discretised) as context for Thompson Sampling.
"""

from __future__ import annotations

import numpy as np
from typing_extensions import override

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
)
from rl_health_interventions.data.normwear_encoder import NormWearEncoder
from rl_health_interventions.data.sensor_signals import generate_sensor_signals
from rl_health_interventions.state import StateView


class NormWearBanditAgent(ContextualBanditAgent):
    """Bandit agent using NormWear FM embeddings as context.

    Pipeline:
      1. Generate synthetic PPG + accelerometer signals from state
      2. Feed through NormWear encoder -> 768-dim [CLS] embedding
      3. Discretise embedding into context key
      4. Thompson Sampling with per-context posteriors
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        seed: int = 42,
        fm_n_bins: int = 8,
        device: str | None = None,
    ) -> None:
        super().__init__(actions=actions, seed=seed)
        self.fm_n_bins = fm_n_bins
        self._encoder = NormWearEncoder(device=device)
        self._rng = np.random.default_rng(seed + 2000)
        # Per-context action counts and reward sums (Thompson Sampling)
        self._counts: dict[tuple[str, ...], dict[str, int]] = {}
        self._rewards: dict[tuple[str, ...], dict[str, float]] = {}
        self._alpha_prior = 1.0
        self._beta_prior = 1.0

    def _ensure_context(self, key: tuple[str, ...]) -> None:
        if key not in self._counts:
            self._counts[key] = dict.fromkeys(self._actions, 0)
            self._rewards[key] = dict.fromkeys(self._actions, 0.0)

    def _get_fm_context(self, state: StateView) -> tuple[str, ...]:
        """Get FM embedding context key from state."""
        # Generate synthetic sensor signals
        signals = generate_sensor_signals(
            state.factor_values,
            seed=int(self._rng.integers(0, 2**31)),
        )

        # Get NormWear embedding
        embedding = self._encoder.encode(signals)  # [1, 768]
        emb = embedding[0]  # [768]

        # Discretise: quantise each dim into bins
        # Use only first 32 dims for manageable context space
        dims_to_use = min(32, len(emb))
        bins = np.digitize(
            emb[:dims_to_use],
            bins=np.linspace(-2, 2, self.fm_n_bins + 1)[1:-1],
        )
        return tuple(str(b) for b in bins)

    @override
    def select_action(self, state: StateView) -> str:
        ctx_key = self._get_fm_context(state)
        self._cached_ctx_key = ctx_key
        self._ensure_context(ctx_key)

        # Thompson Sampling
        samples = {}
        for action in self._actions:
            alpha = self._alpha_prior + self._rewards[ctx_key][action]
            beta = (
                self._beta_prior
                + self._counts[ctx_key][action]
                - self._rewards[ctx_key][action]
            )
            samples[action] = self._rng.beta(max(alpha, 0.01), max(beta, 0.01))

        return max(samples, key=samples.get)  # type: ignore[arg-type]

    @override
    def update(
        self, state: StateView, action: str, reward: float, next_state: StateView
    ) -> None:
        ctx_key = self._cached_ctx_key
        self._ensure_context(ctx_key)
        self._counts[ctx_key][action] += 1
        self._rewards[ctx_key][action] += max(reward, 0.0)
