"""Tests for data preprocessing utilities."""

import pandas as pd
import pytest

from rl_health_interventions.data.preprocessing import (
    add_time_features,
    compute_daily_aggregates,
    resample_to_hourly,
)


@pytest.fixture
def hourly_df() -> pd.DataFrame:
    """Sample hourly DataFrame for testing."""
    return pd.DataFrame(
        {
            "participant_id": ["P001"] * 24 + ["P002"] * 24,
            "time_of_day": pd.date_range("2024-01-01", periods=24, freq="h").tolist()
            * 2,
            "steps": list(range(100, 124)) + list(range(200, 224)),
            "heart_rate": [70.0] * 24 + [65.0] * 24,
        }
    )


@pytest.fixture
def sub_daily_df() -> pd.DataFrame:
    """Sample sub-daily DataFrame (every 15 min) for testing."""
    times = pd.date_range("2024-01-01", "2024-01-02", freq="15min")
    return pd.DataFrame(
        {
            "participant_id": ["P001"] * len(times),
            "time_of_day": times,
            "steps": [10] * len(times),
        }
    )


class TestResampleToHourly:
    """Tests for resample_to_hourly."""

    def test_basic_resample(self, sub_daily_df: pd.DataFrame) -> None:
        result = resample_to_hourly(sub_daily_df, time_col="time_of_day")
        # 15-min data resampled to hourly should have fewer rows
        assert len(result) < len(sub_daily_df)
        assert "time_of_day" in result.columns

    def test_already_hourly(self, hourly_df: pd.DataFrame) -> None:
        result = resample_to_hourly(hourly_df, time_col="time_of_day")
        # Hourly data should stay hourly (minus any all-NaN hours)
        assert len(result) >= 24

    def test_missing_time_column(self) -> None:
        df = pd.DataFrame({"steps": [1, 2, 3]})
        with pytest.raises(ValueError, match="Time column"):
            resample_to_hourly(df, time_col="missing")

    def test_aggregation_mean(self, sub_daily_df: pd.DataFrame) -> None:
        result = resample_to_hourly(
            sub_daily_df, time_col="time_of_day", agg_func={"steps": "sum"}
        )
        assert "steps" in result.columns
        # Sum over 4 fifteen-min intervals = 40 per hour
        assert result["steps"].iloc[0] == 40


class TestComputeDailyAggregates:
    """Tests for compute_daily_aggregates."""

    def test_basic_daily(self, hourly_df: pd.DataFrame) -> None:
        result = compute_daily_aggregates(
            hourly_df, group_col="participant_id", time_col="time_of_day"
        )
        # 2 participants, 1 day each
        assert len(result) == 2
        assert "date" in result.columns
        assert result["participant_id"].nunique() == 2

    def test_steps_are_averaged(self, hourly_df: pd.DataFrame) -> None:
        result = compute_daily_aggregates(
            hourly_df, group_col="participant_id", time_col="time_of_day"
        )
        p1 = result[result["participant_id"] == "P001"]["steps"].iloc[0]
        # Steps should be the mean of 100..123
        assert p1 == pytest.approx(111.5)

    def test_missing_time_column(self) -> None:
        df = pd.DataFrame({"id": ["P1"], "val": [1]})
        with pytest.raises(ValueError, match="Time column"):
            compute_daily_aggregates(df)


class TestAddTimeFeatures:
    """Tests for add_time_features."""

    def test_adds_features(self) -> None:
        df = pd.DataFrame({"time_of_day": pd.to_datetime(["2024-01-01 08:00:00"])})
        result = add_time_features(df, time_col="time_of_day")
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "month" in result.columns
        assert result["hour"].iloc[0] == 8
        assert result["month"].iloc[0] == 1

    def test_weekend_detection(self) -> None:
        # Saturday = day_of_week 5
        df = pd.DataFrame({"time_of_day": pd.to_datetime(["2024-01-06 12:00:00"])})
        result = add_time_features(df, time_col="time_of_day")
        assert result["is_weekend"].iloc[0] == True  # noqa: E712
        assert result["day_of_week"].iloc[0] == 5

    def test_weekday_detection(self) -> None:
        # Monday = day_of_week 0
        df = pd.DataFrame({"time_of_day": pd.to_datetime(["2024-01-01 12:00:00"])})
        result = add_time_features(df, time_col="time_of_day")
        assert result["is_weekend"].iloc[0] == False  # noqa: E712

    def test_missing_time_column(self) -> None:
        df = pd.DataFrame({"val": [1]})
        with pytest.raises(ValueError, match="Time column"):
            add_time_features(df)
