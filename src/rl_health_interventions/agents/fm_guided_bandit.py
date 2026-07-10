"""FM-guided contextual bandit with dynamic action selection.

Two-stage action selection:
1. NormWear embedding → project → cosine similarity → select archetype
2. Within archetype → Thompson Sampling picks specific intervention
3. Action-level epsilon: with prob epsilon, pick random action from all 15
"""

from __future__ import annotations

import numpy as np

from rl_health_interventions.actions.archetypes import (
    ALL_INTERVENTIONS,
    ARCHETYPE_NAMES,
    ARCHETYPES,
)
from rl_health_interventions.actions.fm_selector import FMSelector
from rl_health_interventions.agents._base import Agent
from rl_health_interventions.data.normwear_encoder import NormWearEncoder
from rl_health_interventions.data.sensor_signals import generate_sensor_signals
from rl_health_interventions.state import StateView


class FMGuidedBanditAgent(Agent):
    """Bandit agent with FM-driven archetype selection.

    Pipeline:
      1. State → synthetic sensor signals
      2. Signals → NormWear → 768-dim embedding
      3. Embedding → project → cosine similarity → archetype
      4. Archetype → Thompson Sampling → specific intervention
      5. With prob epsilon: random action from all 15 (exploration)
    """

    def __init__(
        self,
        seed: int = 42,
        device: str | None = None,
        epsilon: float = 0.1,
        selector_lr: float = 0.01,
    ) -> None:
        self._actions = [i.name for i in ALL_INTERVENTIONS]
        self._rng = np.random.default_rng(seed)
        self._epsilon = epsilon

        self._encoder = NormWearEncoder(device=device)
        self._selector = FMSelector(
            embed_dim=768, proj_dim=32, lr=selector_lr, seed=seed
        )

        # Thompson Sampling per-intervention
        self._counts: dict[str, int] = dict.fromkeys(self._actions, 0)
        self._rewards: dict[str, float] = dict.fromkeys(self._actions, 0.0)
        self._alpha_prior = 1.0
        self._beta_prior = 1.0

        # Cached embedding for update
        self._cached_embedding: np.ndarray | None = None
        self._cached_archetype_idx: int | None = None

    def _get_fm_embedding(self, state: StateView) -> np.ndarray:
        """Get NormWear embedding from state."""
        signals = generate_sensor_signals(
            state.factor_values,
            seed=int(self._rng.integers(0, 2**31)),
        )
        return self._encoder.encode(signals)[0]  # [768]

    def _select_intervention(self, archetype_name: str) -> str:
        """Thompson Sampling within the selected archetype."""
        interventions = ARCHETYPES[archetype_name]
        names = [i.name for i in interventions]

        samples = {}
        for name in names:
            alpha = self._alpha_prior + self._rewards[name]
            beta = self._beta_prior + self._counts[name] - self._rewards[name]
            samples[name] = self._rng.beta(max(alpha, 0.01), max(beta, 0.01))

        return max(samples, key=samples.get)  # type: ignore[arg-type]

    def select_action(self, state: StateView) -> str:  # type: ignore[override]
        # Action-level epsilon-exploration
        if self._rng.random() < self._epsilon:
            self._cached_embedding = None
            self._cached_archetype_idx = None
            return self._rng.choice(self._actions)

        embedding = self._get_fm_embedding(state)
        self._cached_embedding = embedding

        # Stage 1: FM → archetype
        archetype_idx, _projected = self._selector.select_archetype(embedding)
        self._cached_archetype_idx = archetype_idx
        archetype_name = ARCHETYPE_NAMES[archetype_idx]

        # Stage 2: Thompson Sampling within archetype
        return self._select_intervention(archetype_name)

    def update(
        self, _state: StateView, action: str, reward: float, _next_state: StateView
    ) -> None:
        # Update bandit posteriors
        self._counts[action] += 1
        self._rewards[action] += max(reward, 0.0)

        # Update FM selector (reinforce archetype vector)
        if (
            self._cached_embedding is not None
            and self._cached_archetype_idx is not None
        ):
            _, projected = self._selector.select_archetype(self._cached_embedding)
            self._selector.update(self._cached_archetype_idx, projected, reward)
