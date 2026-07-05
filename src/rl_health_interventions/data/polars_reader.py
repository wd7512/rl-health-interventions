from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


def _read_csv(path: str, **kwargs: Any) -> pl.DataFrame:
    """Materialise a CSV file inside a thread, so timeout covers I/O."""
    return pl.scan_csv(path, **kwargs).collect()  # type: ignore


def _read_parquet(path: str, **kwargs: Any) -> pl.DataFrame:
    """Materialise a Parquet file inside a thread, so timeout covers I/O."""
    return pl.scan_parquet(path, **kwargs).collect()  # type: ignore


def scan_csv_with_timeout(
    path: str, timeout_s: int = 30, **kwargs: Any
) -> pl.DataFrame:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_read_csv, path, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout:
            raise TimeoutError(
                f"CSV scan timed out after {timeout_s}s: {path}"
            ) from None


def scan_parquet_with_timeout(
    path: str, timeout_s: int = 60, **kwargs: Any
) -> pl.DataFrame:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_read_parquet, path, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout:
            raise TimeoutError(
                f"Parquet scan timed out after {timeout_s}s: {path}"
            ) from None
