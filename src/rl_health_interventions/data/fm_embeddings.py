"""Synthetic FM embedding generator for proof-of-concept experiments.

Generates dense vector representations of health state that simulate what a
real wearable foundation model (e.g. NormWear, SensorFM) would produce from
sensor data. Each embedding encodes physiological information in a continuous
vector space.

This is a PLACEHOLDER — replace with real FM inference when weights are available.
The architecture (FM embeddings → context features → bandit agent) is the same
regardless of whether the embedding comes from a synthetic generator or NormWear.
"""

from __future__ import annotations

import numpy as np

from rl_health_interventions.state import StateView

# Embedding dimension (matches NormWear's hidden size / 4 for a lightweight POC)
EMBED_DIM = 16

# Deterministic mapping from state factor values to embedding components.
# Each factor contributes a fixed offset; noise simulates inter-individual
# variability that a real FM would capture from raw sensor data.
_FACTOR_OFFSETS: dict[str, dict[str, np.ndarray]] = {
    "step_bin": {
        "inactive": np.array(
            [
                -0.8,
                -0.2,
                0.1,
                0.3,
                -0.5,
                0.0,
                0.2,
                -0.3,
                0.1,
                -0.4,
                0.6,
                -0.1,
                0.3,
                -0.2,
                0.0,
                0.5,
            ]
        ),
        "moderate": np.array(
            [
                0.0,
                0.3,
                -0.1,
                0.2,
                0.1,
                -0.2,
                0.4,
                0.0,
                -0.3,
                0.2,
                -0.1,
                0.3,
                -0.1,
                0.4,
                -0.2,
                0.1,
            ]
        ),
        "active": np.array(
            [
                0.6,
                0.5,
                -0.3,
                -0.1,
                0.4,
                0.3,
                -0.2,
                0.5,
                0.2,
                -0.1,
                -0.4,
                0.2,
                -0.3,
                0.1,
                0.3,
                -0.4,
            ]
        ),
    },
    "sleep": {
        "good": np.array(
            [
                0.3,
                -0.1,
                0.5,
                0.2,
                -0.3,
                0.4,
                -0.1,
                0.3,
                0.1,
                -0.2,
                0.3,
                -0.4,
                0.2,
                0.0,
                -0.1,
                0.2,
            ]
        ),
        "poor": np.array(
            [
                -0.3,
                0.1,
                -0.5,
                -0.2,
                0.3,
                -0.4,
                0.1,
                -0.3,
                -0.1,
                0.2,
                -0.3,
                0.4,
                -0.2,
                0.0,
                0.1,
                -0.2,
            ]
        ),
    },
    "day_of_week": {
        "weekday": np.array(
            [
                0.1,
                0.0,
                -0.1,
                0.0,
                0.1,
                -0.1,
                0.0,
                0.1,
                0.0,
                -0.1,
                0.1,
                0.0,
                -0.1,
                0.0,
                0.1,
                0.0,
            ]
        ),
        "weekend": np.array(
            [
                -0.1,
                0.0,
                0.1,
                0.0,
                -0.1,
                0.1,
                0.0,
                -0.1,
                0.0,
                0.1,
                -0.1,
                0.0,
                0.1,
                0.0,
                -0.1,
                0.0,
            ]
        ),
    },
    "burden": {
        "low": np.array(
            [
                0.2,
                0.1,
                -0.1,
                0.0,
                0.2,
                -0.1,
                0.1,
                0.0,
                -0.1,
                0.2,
                -0.1,
                0.0,
                0.1,
                -0.1,
                0.0,
                0.1,
            ]
        ),
        "medium": np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        "high": np.array(
            [
                -0.2,
                -0.1,
                0.1,
                0.0,
                -0.2,
                0.1,
                -0.1,
                0.0,
                0.1,
                -0.2,
                0.1,
                0.0,
                -0.1,
                0.1,
                0.0,
                -0.1,
            ]
        ),
    },
}

# Per-factor noise scale. Real FMs would capture inter-individual variability.
_NOISE_SCALE = 0.05


def generate_fm_embedding(
    state: StateView,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a synthetic FM embedding from the current state.

    In production, this function would:
      1. Collect raw sensor data for the current time window
      2. Feed it through NormWear/SensorFM encoder
      3. Return the latent vector

    For the POC, we deterministically combine state factor offsets + noise.
    The key property: same state → same embedding (up to noise), different
    states → different embeddings. This mimics what a real FM does.
    """
    if rng is None:
        rng = np.random.default_rng()

    embedding = np.zeros(EMBED_DIM, dtype=np.float64)
    for factor_name, value in state.factor_values.items():
        if factor_name in _FACTOR_OFFSETS and value in _FACTOR_OFFSETS[factor_name]:
            embedding += _FACTOR_OFFSETS[factor_name][value]

    # Add noise (simulates raw sensor variability)
    embedding += rng.normal(0, _NOISE_SCALE, size=EMBED_DIM)

    # L2-normalise (mimics what contrastive FMs do)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    return embedding


def embedding_to_context_key(
    embedding: np.ndarray,
    n_bins: int = 4,
) -> tuple[str, ...]:
    """Discretise an FM embedding into a hashable context key.

    This is the POC approach — quantise the embedding into bins so bandit
    agents can use it as a context key. In production, you'd use:
      - Linear contextual bandits (continuous context, no discretisation)
      - Neural contextual bandits (embedding as input to a neural net)

    The discretisation proves the concept: FM embeddings create a richer
    context space than hand-crafted factors alone.
    """
    # Quantise each dimension into n_bins
    bins = np.digitize(embedding, bins=np.linspace(-1.5, 1.5, n_bins + 1)[1:-1])
    return tuple(str(b) for b in bins)
