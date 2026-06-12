"""Tests for 3-fold cross-validation (Task 3.1)."""

import numpy as np
import pytest

from paper_reproduction.simulation.cross_validation import create_folds


class TestCreateFolds:
    def test_three_iterations(self):
        folds = create_folds(30, n_folds=3, rng=np.random.default_rng(42))
        assert len(folds) == 3

    def test_each_participant_in_one_test_fold(self):
        folds = create_folds(30, n_folds=3, rng=np.random.default_rng(42))
        counts = np.zeros(30, dtype=int)
        for _, test_idx in folds:
            counts[test_idx] += 1
        assert np.all(counts == 1)

    def test_train_test_disjoint(self):
        folds = create_folds(30, n_folds=3, rng=np.random.default_rng(42))
        for train_idx, test_idx in folds:
            train_set = set(train_idx.tolist())
            test_set = set(test_idx.tolist())
            assert len(train_set & test_set) == 0

    def test_all_participants_covered(self):
        n = 30
        folds = create_folds(n, n_folds=3, rng=np.random.default_rng(42))
        all_test = np.concatenate([test for _, test in folds])
        assert sorted(all_test.tolist()) == list(range(n))

    def test_fold_sizes_roughly_equal(self):
        folds = create_folds(31, n_folds=3, rng=np.random.default_rng(42))
        sizes = [len(test) for _, test in folds]
        assert max(sizes) - min(sizes) <= 1

    def test_deterministic_with_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        folds1 = create_folds(30, n_folds=3, rng=rng1)
        folds2 = create_folds(30, n_folds=3, rng=rng2)
        for (t1, _), (t2, _) in zip(folds1, folds2):
            np.testing.assert_array_equal(t1, t2)

    def test_different_seeds_different_splits(self):
        folds1 = create_folds(30, n_folds=3, rng=np.random.default_rng(42))
        folds2 = create_folds(30, n_folds=3, rng=np.random.default_rng(99))
        # At least one fold should differ
        same = all(
            np.array_equal(t1, t2)
            for (t1, _), (t2, _) in zip(folds1, folds2)
        )
        assert not same
