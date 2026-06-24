"""Tests for NHANES data loaders and simulation utilities."""

from __future__ import annotations

import numpy as np

from rl_health_interventions.data.nhanes import SyntheticNHANESGenerator, NHANESLoader
from rl_health_interventions.simulation.cross_validation import create_folds
from rl_health_interventions.simulation.results import (
    SimulationResults,
    CVFoldResult,
    ParticipantResult,
)


class TestSyntheticNHANESGenerator:
    def test_generate_shape(self) -> None:
        gen = SyntheticNHANESGenerator(seed=42)
        data = gen.generate(n_participants=5, n_days=7, n_windows=10)
        assert data.shape == (5, 7, 10)

    def test_non_negative(self) -> None:
        gen = SyntheticNHANESGenerator(seed=42)
        data = gen.generate(n_participants=3, n_days=5, n_windows=10)
        assert np.all(data >= 0)

    def test_reproducible(self) -> None:
        gen1 = SyntheticNHANESGenerator(seed=42)
        gen2 = SyntheticNHANESGenerator(seed=42)
        d1 = gen1.generate(n_participants=2, n_days=3)
        d2 = gen2.generate(n_participants=2, n_days=3)
        np.testing.assert_array_equal(d1, d2)


class TestNHANESLoader:
    def test_synthetic_dispatch(self) -> None:
        loader = NHANESLoader(
            data_source="synthetic", n_participants=3, n_days=5, seed=42
        )
        data, meta = loader.load()
        assert data.shape == (3, 5, 10)
        assert len(meta) == 3

    def test_unknown_source_raises(self) -> None:
        loader = NHANESLoader(data_source="unknown")
        try:
            loader.load()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestCreateFolds:
    def test_number_of_folds(self) -> None:
        folds = create_folds(n_participants=9, n_folds=3, rng=np.random.default_rng(42))
        assert len(folds) == 3

    def test_all_participants_covered(self) -> None:
        folds = create_folds(n_participants=9, n_folds=3, rng=np.random.default_rng(42))
        all_test = np.concatenate([test for _, test in folds])
        assert len(np.unique(all_test)) == 9

    def test_train_test_disjoint(self) -> None:
        folds = create_folds(n_participants=9, n_folds=3, rng=np.random.default_rng(42))
        for train, test in folds:
            assert len(np.intersect1d(train, test)) == 0


class TestSimulationResults:
    def test_compute_summary(self) -> None:
        pr1 = ParticipantResult(0, 100.0, 80.0, 20.0)
        pr2 = ParticipantResult(1, 70.0, 90.0, -20.0)
        cv = CVFoldResult(0, [0, 1], [2], 0.9, 0.5, [pr1, pr2])
        results = SimulationResults(config={}, cv_results=[cv])
        summary = results.compute_summary()
        assert summary["n_participants"] == 2
        assert summary["n_improved"] == 1
        assert summary["pct_improved"] == 50.0
