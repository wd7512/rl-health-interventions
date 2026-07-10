"""Contextual bandit agent that uses FM embeddings as context features.

This is the POC agent — it generates a synthetic FM embedding from the
current state, discretises it, and uses the resulting key for per-context
parameter lookup. Same architecture as the existing contextual bandits,
but with a richer context space derived from FM embeddings.
"""

from __future__ import annotations

import numpy as np
from typing_extensions import override

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
)
from rl_health_interventions.data.fm_embeddings import (
    embedding_to_context_key,
    generate_fm_embedding,
)
from rl_health_interventions.state import StateView


class FMContextualBanditAgent(ContextualBanditAgent):
    """Bandit agent that uses FM embeddings as context.

    Unlike the base ContextualBanditAgent which reads discrete factor values
    from StateView, this agent:
      1. Generates an FM embedding from the full state
      2. Discretises it into a context key
      3. Uses that key for per-action parameter lookup

    The context space is much richer than hand-crafted factors:
      - 4 state factors x 2-3 values each = ~36 contexts
      - FM embedding with 16 dims x 4 bins each = 4^16 = ~4B contexts
        (in practice, discretised to ~50-200 active contexts)

    This proves the architecture works. With a real FM, the embedding
    would capture physiological patterns invisible to hand-crafted factors.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        seed: int = 42,
        fm_n_bins: int = 4,
    ) -> None:
        super().__init__(actions=actions, seed=seed)
        self.fm_n_bins = fm_n_bins
        self._fm_rng = np.random.default_rng(seed + 1000)
        # Per-context action counts and reward sums (Thompson Sampling style)
        self._counts: dict[tuple[str, ...], dict[str, int]] = {}
        self._rewards: dict[tuple[str, ...], dict[str, float]] = {}
        self._alpha_prior = 1.0
        self._beta_prior = 1.0

    def _ensure_context(self, key: tuple[str, ...]) -> None:
        if key not in self._counts:
            self._counts[key] = dict.fromkeys(self._actions, 0)
            self._rewards[key] = dict.fromkeys(self._actions, 0.0)

    @override
    def select_action(self, state: StateView) -> str:
        embedding = generate_fm_embedding(state, rng=self._fm_rng)
        ctx_key = embedding_to_context_key(embedding, n_bins=self.fm_n_bins)
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
        embedding = generate_fm_embedding(state, rng=self._fm_rng)
        ctx_key = embedding_to_context_key(embedding, n_bins=self.fm_n_bins)
        self._ensure_context(ctx_key)
        self._counts[ctx_key][action] += 1
        self._rewards[ctx_key][action] += max(reward, 0.0)
