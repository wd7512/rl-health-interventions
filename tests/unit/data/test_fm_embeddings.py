"""Tests for the FM embedding POC module."""
from __future__ import annotations

import numpy as np
import pytest

from rl_health_interventions.data.fm_embeddings import (
    EMBED_DIM,
    embedding_to_context_key,
    generate_fm_embedding,
)
from rl_health_interventions.state import StateView


class TestGenerateFMEmbedding:
    def test_output_shape(self) -> None:
        state = StateView({"step_bin": "inactive", "sleep": "good"})
        rng = np.random.default_rng(42)
        embedding = generate_fm_embedding(state, rng=rng)
        assert embedding.shape == (EMBED_DIM,)

    def test_normalised(self) -> None:
        state = StateView({"step_bin": "active", "sleep": "poor"})
        rng = np.random.default_rng(42)
        embedding = generate_fm_embedding(state, rng=rng)
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-6

    def test_deterministic_with_same_seed(self) -> None:
        state = StateView({"step_bin": "moderate", "sleep": "good"})
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        e1 = generate_fm_embedding(state, rng=rng1)
        e2 = generate_fm_embedding(state, rng=rng2)
        np.testing.assert_array_equal(e1, e2)

    def test_different_states_produce_different_embeddings(self) -> None:
        rng = np.random.default_rng(42)
        state_a = StateView({"step_bin": "inactive", "sleep": "poor"})
        state_b = StateView({"step_bin": "active", "sleep": "good"})
        e_a = generate_fm_embedding(state_a, rng=rng)
        e_b = generate_fm_embedding(state_b, rng=rng)
        assert not np.allclose(e_a, e_b, atol=0.1)


class TestEmbeddingToContextKey:
    def test_output_is_tuple_of_strings(self) -> None:
        embedding = np.random.default_rng(42).normal(size=EMBED_DIM)
        key = embedding_to_context_key(embedding, n_bins=4)
        assert isinstance(key, tuple)
        assert len(key) == EMBED_DIM
        assert all(isinstance(v, str) for v in key)

    def test_same_embedding_same_key(self) -> None:
        embedding = np.random.default_rng(42).normal(size=EMBED_DIM)
        key1 = embedding_to_context_key(embedding, n_bins=4)
        key2 = embedding_to_context_key(embedding, n_bins=4)
        assert key1 == key2

    def test_different_n_bins_different_keys(self) -> None:
        embedding = np.random.default_rng(42).normal(size=EMBED_DIM)
        key4 = embedding_to_context_key(embedding, n_bins=4)
        key8 = embedding_to_context_key(embedding, n_bins=8)
        # More bins = more resolution = different key
        assert key4 != key8
