"""Tests for NHANES data ingestion (Task 2.1).

Tests SyntheticNHANESGenerator data properties and NHANESLoader interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paper_reproduction.data.nhanes_loader import (
    NHANESLoader,
    RealNHANESLoader,
    SyntheticNHANESGenerator,
)


class TestSyntheticNHANESGeneratorOutput:
    def test_output_shape_default(self):
        """Default call produces (100, 7, 10)."""
        gen = SyntheticNHANESGenerator(seed=42)
        data = gen.generate()
        assert data.shape == (100, 7, 10)

    def test_output_shape_custom(self):
        """Custom dimensions produce correct shape."""
        gen = SyntheticNHANESGenerator(seed=42)
        data = gen.generate(n_participants=50, n_days=14, n_windows=5)
        assert data.shape == (50, 14, 5)

    def test_single_participant(self):
        """Single participant still yields 3D array."""
        gen = SyntheticNHANESGenerator(seed=42)
        data = gen.generate(n_participants=1, n_days=7, n_windows=10)
        assert data.shape == (1, 7, 10)

    def test_single_day(self):
        """Single day produces correct shape."""
        gen = SyntheticNHANESGenerator(seed=42)
        data = gen.generate(n_participants=10, n_days=1, n_windows=10)
        assert data.shape == (10, 1, 10)


class TestStepCountProperties:
    def test_non_negative(self):
        """All step counts are non-negative."""
        gen = SyntheticNHANESGenerator(seed=42)
        data = gen.generate(n_participants=50, n_days=7, n_windows=10)
        assert np.all(data >= 0)

    def test_no_nan(self):
        """No NaN values in output."""
        gen = SyntheticNHANESGenerator(seed=42)
        data = gen.generate(n_participants=50, n_days=7, n_windows=10)
        assert not np.any(np.isnan(data))

    def test_integer_values(self):
        """Step counts are integers."""
        gen = SyntheticNHANESGenerator(seed=42)
        data = gen.generate(n_participants=20, n_days=7, n_windows=10)
        assert np.all(data == np.round(data))

    def test_daily_total_range(self):
        """Daily totals are roughly 1000-12000 for most participants."""
        gen = SyntheticNHANESGenerator(seed=42)
        data = gen.generate(n_participants=100, n_days=7, n_windows=10)
        daily = data.sum(axis=-1)  # (100, 7)
        mean_daily = daily.mean()
        assert 2000 < mean_daily < 10000


class TestTimeOfDayPatterns:
    def test_morning_greater_than_late_night(self):
        """Morning peak (windows 2-3) > late night (windows 0, 9)."""
        gen = SyntheticNHANESGenerator(seed=42)
        data = gen.generate(n_participants=100, n_days=7, n_windows=10)
        morning = data[:, :, 2:4].mean()
        night = data[:, :, [0, 9]].mean()
        assert morning > night * 1.2  # at least 20% higher

    def test_midday_decline_vs_morning_peak(self):
        """Afternoon windows (4-6) lower than morning peak (2-3)."""
        gen = SyntheticNHANESGenerator(seed=42)
        data = gen.generate(n_participants=100, n_days=7, n_windows=10)
        morning = data[:, :, 2:4].mean()
        afternoon = data[:, :, 4:7].mean()
        assert afternoon < morning


class TestDeterminism:
    def test_same_seed_identical(self):
        """Same seed produces identical data."""
        gen1 = SyntheticNHANESGenerator(seed=42)
        gen2 = SyntheticNHANESGenerator(seed=42)
        d1 = gen1.generate(n_participants=30, n_days=7, n_windows=10)
        d2 = gen2.generate(n_participants=30, n_days=7, n_windows=10)
        np.testing.assert_array_equal(d1, d2)

    def test_different_seed_different(self):
        """Different seeds produce different data."""
        gen1 = SyntheticNHANESGenerator(seed=42)
        gen2 = SyntheticNHANESGenerator(seed=99)
        d1 = gen1.generate(n_participants=30, n_days=7, n_windows=10)
        d2 = gen2.generate(n_participants=30, n_days=7, n_windows=10)
        assert not np.allclose(d1, d2)

    def test_reproducible_across_calls(self):
        """Two calls with same generator produce same result after reset."""
        gen = SyntheticNHANESGenerator(seed=42)
        d1 = gen.generate(n_participants=10, n_days=7, n_windows=10)

        gen2 = SyntheticNHANESGenerator(seed=42)
        d2 = gen2.generate(n_participants=10, n_days=7, n_windows=10)
        np.testing.assert_array_equal(d1, d2)


class TestNHANESLoaderInterface:
    def test_loader_synthetic_default(self):
        """NHANESLoader with synthetic source returns correct type."""
        loader = NHANESLoader(
            data_source="synthetic", n_participants=20, n_days=7, seed=42
        )
        data = loader.load()
        assert isinstance(data, np.ndarray)
        assert data.shape == (20, 7, 10)

    def test_loader_synthetic_deterministic(self):
        """NHANESLoader is deterministic with same seed."""
        l1 = NHANESLoader(
            data_source="synthetic", n_participants=15, n_days=7, seed=123
        )
        l2 = NHANESLoader(
            data_source="synthetic", n_participants=15, n_days=7, seed=123
        )
        np.testing.assert_array_equal(l1.load(), l2.load())

    def test_loader_real_no_path_raises(self):
        """NHANESLoader with real source and no data_path raises ValueError."""
        loader = NHANESLoader(data_source="real")
        with pytest.raises(ValueError, match="data_path"):
            loader.load()

    def test_loader_real_with_path(self, small_nhanes_csv):
        """NHANESLoader dispatches to RealNHANESLoader with data_path."""
        loader = NHANESLoader(
            data_source="real",
            n_participants=3,
            n_days=5,
            seed=42,
            data_path=small_nhanes_csv,
        )
        data = loader.load()
        assert isinstance(data, np.ndarray)
        assert data.shape == (3, 5, 10)


class TestRealNHANESLoader:
    """Tests for RealNHANESLoader with a small synthetic CSV."""

    def test_loader_shape(self, small_nhanes_csv):
        """RealNHANESLoader returns correct shape."""
        loader = RealNHANESLoader(
            data_path=small_nhanes_csv,
            n_participants=3,
            n_days=5,
            n_windows=10,
            seed=42,
        )
        data = loader.load()
        assert data.shape == (3, 5, 10)

    def test_non_negative(self, small_nhanes_csv):
        """All step counts are non-negative."""
        loader = RealNHANESLoader(
            data_path=small_nhanes_csv,
            n_participants=3,
            n_days=5,
            n_windows=10,
            seed=42,
        )
        data = loader.load()
        assert np.all(data >= 0)

    def test_no_nan(self, small_nhanes_csv):
        """No NaN values in output."""
        loader = RealNHANESLoader(
            data_path=small_nhanes_csv,
            n_participants=3,
            n_days=5,
            n_windows=10,
            seed=42,
        )
        data = loader.load()
        assert not np.any(np.isnan(data))

    def test_deterministic_same_seed(self, small_nhanes_csv):
        """Same seed produces identical results."""
        l1 = RealNHANESLoader(
            data_path=small_nhanes_csv,
            n_participants=3,
            n_days=5,
            n_windows=10,
            seed=42,
        )
        l2 = RealNHANESLoader(
            data_path=small_nhanes_csv,
            n_participants=3,
            n_days=5,
            n_windows=10,
            seed=42,
        )
        np.testing.assert_array_equal(l1.load(), l2.load())

    def test_missing_file_fallback(self, tmp_path):
        """Non-existent path falls back to synthetic data."""
        fake_path = str(tmp_path / "nonexistent.csv")
        loader = RealNHANESLoader(
            data_path=fake_path,
            n_participants=10,
            n_days=7,
            n_windows=10,
            seed=42,
        )
        data = loader.load()
        assert data.shape == (10, 7, 10)
        assert np.all(data >= 0)

    def test_partial_data(self, tmp_path):
        """Loader handles participants with fewer days than requested."""
        filepath = tmp_path / "partial.csv"
        rng = np.random.default_rng(42)

        rows = []
        for participant in [1, 2]:
            for day in range(1, 6):
                row_data = {
                    "SEQN": participant,
                    "PAXDAYM": day,
                    "PAXDAYWM": (day % 7) + 1,
                }
                for minute in range(1, 1441):
                    row_data[f"min_{minute}"] = max(0, int(rng.poisson(5)))
                rows.append(row_data)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

        loader = RealNHANESLoader(
            data_path=str(filepath),
            n_participants=2,
            n_days=5,
            n_windows=10,
            seed=42,
        )
        data = loader.load()
        assert data.shape == (2, 5, 10)

    def test_insufficient_valid_days(self, tmp_path):
        """Participant with insufficient valid days is skipped."""
        filepath = tmp_path / "insufficient.csv"
        rng = np.random.default_rng(42)

        rows = []
        for participant in [1, 2, 3]:
            for day in range(1, 4):
                row_data = {
                    "SEQN": participant,
                    "PAXDAYM": day,
                    "PAXDAYWM": (day % 7) + 1,
                }
                for minute in range(1, 1441):
                    row_data[f"min_{minute}"] = max(0, int(rng.poisson(5)))
                rows.append(row_data)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

        loader = RealNHANESLoader(
            data_path=str(filepath),
            n_participants=3,
            n_days=5,
            n_windows=10,
            seed=42,
        )
        data = loader.load()
        assert data.shape[0] < 3  # fewer participants than requested

    def test_min_valid_minutes_filter(self, tmp_path):
        """Loader respects min_valid_minutes threshold."""
        filepath = tmp_path / "valid_filter.csv"
        rng = np.random.default_rng(42)

        rows = []
        for participant in [1, 2]:
            for day in range(1, 3):
                row_data = {
                    "SEQN": participant,
                    "PAXDAYM": day,
                    "PAXDAYWM": (day % 7) + 1,
                }
                for minute in range(1, 1441):
                    row_data[f"min_{minute}"] = max(0, int(rng.poisson(5)))
                rows.append(row_data)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

        loader = RealNHANESLoader(
            data_path=str(filepath),
            n_participants=2,
            n_days=2,
            n_windows=10,
            min_valid_minutes=1440,
            seed=42,
        )
        # All have 1440 valid minutes, so should work
        data = loader.load()
        assert data.shape == (2, 2, 10)


@pytest.fixture
def small_nhanes_csv(tmp_path):
    """Create a minimal NHANES CSV for testing RealNHANESLoader."""
    filepath = tmp_path / "test_nhanes.csv"
    rng = np.random.default_rng(42)

    rows: list[dict[str, object]] = []
    for participant in [100, 200, 300]:
        for day in range(1, 6):
            row_data: dict[str, object] = {
                "SEQN": participant,
                "PAXDAYM": day,
                "PAXDAYWM": (day % 7) + 1,
            }
            for minute in range(1, 1441):
                if minute <= 120 and day == 5:
                    row_data[f"min_{minute}"] = None
                else:
                    row_data[f"min_{minute}"] = max(0, int(rng.poisson(5)))
            rows.append(row_data)

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return str(filepath)
