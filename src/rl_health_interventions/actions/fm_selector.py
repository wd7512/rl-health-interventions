from __future__ import annotations

import numpy as np

from rl_health_interventions.actions.archetypes import ARCHETYPE_NAMES


class FMSelector:
    """Project FM embedding → low-dim → cosine similarity to 3 archetype vectors."""

    def __init__(
        self,
        embed_dim: int = 16,
        proj_dim: int = 32,
        n_archetypes: int = 3,
        lr: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.proj_dim = proj_dim
        self.lr = lr
        self.n_archetypes = n_archetypes
        self._rng = np.random.default_rng(seed)

        self.W = self._rng.normal(0, 0.01, size=(embed_dim, proj_dim)).astype(
            np.float64
        )
        self.archetype_vectors = self._rng.normal(
            0, 0.1, size=(n_archetypes, proj_dim)
        ).astype(np.float64)
        self._normalise_archetypes()

    def _normalise_archetypes(self) -> None:
        norms = np.linalg.norm(self.archetype_vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.archetype_vectors /= norms

    def project(self, embedding: np.ndarray) -> np.ndarray:
        return embedding @ self.W

    def select_archetype(self, embedding: np.ndarray) -> tuple[int, np.ndarray]:
        projected = self.project(embedding)
        similarities = self.archetype_vectors @ projected
        archetype_idx = int(np.argmax(similarities))
        return archetype_idx, projected

    def update(
        self, archetype_idx: int, projected_emb: np.ndarray, reward: float
    ) -> None:
        self.archetype_vectors[archetype_idx] += self.lr * reward * projected_emb
        self._normalise_archetypes()

    @property
    def archetype_names(self) -> list[str]:
        return ARCHETYPE_NAMES[: self.n_archetypes]
