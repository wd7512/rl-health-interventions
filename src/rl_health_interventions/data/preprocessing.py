"""Feature engineering for wearable health data."""

from __future__ import annotations

import pandas as pd


def resample_to_hourly(
    df: pd.DataFrame,
    time_col: str = "time_of_day",
    agg_func: str | dict | None = None,
) -> pd.DataFrame:
    """Resample data to hourly intervals.

    Args:
        df: DataFrame with a datetime column.
        time_col: Name of the datetime column.
        agg_func: Aggregation function(s). Defaults to mean for numeric,
                  first for non-numeric.

    Returns:
        Resampled DataFrame with one row per hour.
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)

    if agg_func is None:
        # Default: mean for numeric, first for others
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()
        agg_func = {}
        for col in numeric_cols:
            agg_func[col] = "mean"
        for col in non_numeric_cols:
            agg_func[col] = "first"

    resampled = df.resample("h").agg(agg_func)
    resampled = resampled.dropna(how="all")
    return resampled.reset_index()


def compute_daily_aggregates(
    df: pd.DataFrame,
    group_col: str = "participant_id",
    time_col: str = "time_of_day",
) -> pd.DataFrame:
    """Compute daily aggregates from hourly or sub-daily data.

    Args:
        df: DataFrame with datetime and participant columns.
        group_col: Column identifying participants.
        time_col: Name of the datetime column.

    Returns:
        DataFrame with one row per participant per day.
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["date"] = df[time_col].dt.date

    # Build aggregation dict: mean for numeric, first for categorical
    agg_dict = {}
    for col in df.columns:
        if col in (group_col, time_col, "date"):
            continue
        if df[col].dtype in ("float64", "int64", "Int64"):
            agg_dict[col] = "mean"
        else:
            agg_dict[col] = "first"

    daily = df.groupby([group_col, "date"]).agg(agg_dict).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def add_time_features(
    df: pd.DataFrame,
    time_col: str = "time_of_day",
) -> pd.DataFrame:
    """Add temporal features from a datetime column.

    Adds: hour, day_of_week, is_weekend, month.

    Args:
        df: DataFrame with a datetime column.
        time_col: Name of the datetime column.

    Returns:
        DataFrame with additional time features.
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found")

    df = df.copy()
    dt = pd.to_datetime(df[time_col])

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = dt.dt.dayofweek >= 5
    df["month"] = dt.dt.month

    return df
