from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from rl_health_interventions.data.polars_reader import (
    scan_csv_with_timeout,
    scan_parquet_with_timeout,
)


def test_scan_csv_with_timeout_success() -> None:
    """Happy-path: scan a small CSV file within the timeout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("col1,col2\n1,2\n3,4\n")
        path = f.name

    df = scan_csv_with_timeout(path, timeout_s=10)
    assert df.shape == (2, 2)
    assert df.columns == ["col1", "col2"]
    Path(path).unlink()


def test_scan_parquet_with_timeout_success() -> None:
    """Happy-path: scan a small Parquet file within the timeout."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.write_parquet(path)

    result = scan_parquet_with_timeout(path, timeout_s=10)
    assert result.shape == (2, 2)
    assert result.columns == ["a", "b"]
    Path(path).unlink()
