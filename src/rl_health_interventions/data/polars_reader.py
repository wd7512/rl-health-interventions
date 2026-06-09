from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


def scan_csv_with_timeout(
    path: str, timeout_s: int = 30, **kwargs: Any
) -> pl.DataFrame:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pl.scan_csv, path, **kwargs)
        try:
            return future.result(timeout=timeout_s).collect()  # type: ignore
        except FuturesTimeout:
            raise TimeoutError(f"CSV scan timed out after {timeout_s}s: {path}")


def scan_parquet_with_timeout(
    path: str, timeout_s: int = 60, **kwargs: Any
) -> pl.DataFrame:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pl.scan_parquet, path, **kwargs)
        try:
            return future.result(timeout=timeout_s).collect()  # type: ignore
        except FuturesTimeout:
            raise TimeoutError(f"Parquet scan timed out after {timeout_s}s: {path}")
