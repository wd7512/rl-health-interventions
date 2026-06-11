#!/usr/bin/env python3
"""Run exploratory data analysis across all supported health-intervention datasets.

Usage::

    python scripts/eda_all_datasets.py
    python -m scripts.eda_all_datasets --data-dir /path/to/data

Produces a terminal report and saves a copy to ``data/eda_report.txt``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Any

import polars as pl

from rl_health_interventions.data.loaders import load_all

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────


def _section(title: str, width: int = 78) -> str:
    """Format a section header line."""
    return f"{'─' * 3} {title} {'─' * (width - len(title) - 5)}"


def _fmt_count(val: Any) -> str:
    """Format a count or percentage for alignment."""
    if val is None or (isinstance(val, float) and val != val):  # NaN
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def describe_dataset(
    name: str, df: pl.DataFrame, lines: list[str], width: int = 78
) -> None:
    """Append a full EDA description of *df* to *lines*."""
    lines.append("")
    lines.append(_section(name, width))
    lines.append("")

    # 1. Shape
    rows, cols = df.shape
    lines.append(f"  Shape: {rows:,} rows × {cols} columns")
    lines.append("")

    # 2. Column names & dtypes
    lines.append("  Columns & dtypes:")
    for col_name, dtype in zip(df.columns, df.dtypes):
        lines.append(f"    {col_name}: {dtype}")
    lines.append("")

    # 3. Missing values
    missing = df.null_count()
    total = df.height
    lines.append("  Missing values:")
    has_missing = False
    for col_name in df.columns:
        m = missing[col_name][0] if missing.width > 1 else missing[col_name, 0]
        if m > 0:
            has_missing = True
            pct = m / total * 100
            lines.append(f"    {col_name}: {m:,} ({pct:.2f}%)")
    if not has_missing:
        lines.append("    (none)")
    lines.append("")

    # 4. Descriptive statistics for numeric columns
    numeric_cols = [
        c
        for c, dt in zip(df.columns, df.dtypes)
        if dt in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
    ]
    if numeric_cols:
        lines.append("  Numeric column statistics:")
        for col_name in numeric_cols:
            try:
                stats = df[col_name].describe()
                # describe returns a DataFrame with 'statistic' and 'value' columns
                # In newer polars, describe() returns a DataFrame
                lines.append(f"    {col_name}:")
                for row in stats.iter_rows():
                    stat_name = str(row[0]).rjust(16)
                    stat_val = _fmt_count(row[1])
                    lines.append(f"      {stat_name}: {stat_val}")
            except Exception:
                lines.append(f"    {col_name}: (could not compute)")
        lines.append("")

    # 5. Value counts for (low-cardinality) categorical / string columns
    cat_cols = [
        c
        for c, dt in zip(df.columns, df.dtypes)
        if dt in (pl.Utf8, pl.Categorical, pl.Enum)
    ]
    if cat_cols:
        lines.append("  Categorical column value counts (top 10):")
        for col_name in cat_cols:
            lines.append(f"    {col_name}:")
            try:
                vc = df[col_name].value_counts(sort=True).head(10)
                for row in vc.iter_rows():
                    val = str(row[0])[:40]
                    cnt = row[1]
                    lines.append(f"      {val!r:<42}: {cnt:,}")
            except Exception:
                lines.append("      (could not compute)")
        lines.append("")

    # 6. Temporal coverage
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        try:
            ts_min = ts.min()
            ts_max = ts.max()
        except Exception:
            ts_min = ts_max = "—"
        lines.append(f"  Temporal coverage: {ts_min}  →  {ts_max}")
    if "user_id" in df.columns:
        unique_users = df["user_id"].n_unique()
        lines.append(f"  Unique users: {unique_users:,}")
    lines.append("")

    # 7. Correlation matrix for numeric features (printed)
    if len(numeric_cols) >= 2:
        lines.append("  Correlation matrix (numeric features):")
        try:
            clean = df.select(numeric_cols).drop_nulls()
            if clean.height > 10:
                corr = clean.corr()
                for i, row in enumerate(corr.iter_rows()):
                    col_name = str(row[0]).rjust(20)
                    values = "  ".join(
                        f"{v:7.4f}" if isinstance(v, float) else str(v)[:7]
                        for v in row[1:]
                    )
                    lines.append(f"      {col_name}:  {values}")
            else:
                lines.append("      (too few complete rows)")
        except Exception:
            lines.append("      (could not compute)")
        lines.append("")

    lines.append(f"  {'─' * width}")
    lines.append("")


def summary_table(
    results: dict[str, pl.DataFrame | None], width: int = 78
) -> list[str]:
    """Build a cross-dataset comparison table."""
    lines: list[str] = []
    lines.append("")
    lines.append(_section("Summary Comparison", width))
    lines.append("")
    header = (
        f"  {'Dataset':<20} {'Rows':>12} {'Cols':>5} {'Users':>8} "
        f"{'Missing%':>9} {'NumCols':>8}"
    )
    lines.append(header)
    lines.append("  " + "─" * (len(header) - 2))

    for name, df in results.items():
        if df is None:
            lines.append(f"  {name:<20} {'—':>12} {'—':>5} {'—':>8} {'—':>9} {'—':>8}")
            continue

        rows_count = f"{df.height:,}"
        cols_count = str(df.width)
        users = f"{df['user_id'].n_unique():,}" if "user_id" in df.columns else "—"
        # Total missing percentage
        total_cells = df.height * df.width
        if total_cells > 0:
            missing_total = sum(df.select(pl.all().null_count()).to_numpy().flatten())
            missing_pct = f"{missing_total / total_cells * 100:.1f}%"
        else:
            missing_pct = "—"

        numeric_count = sum(
            1
            for dt in df.dtypes
            if dt in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
        )

        lines.append(
            f"  {name:<20} {rows_count:>12} {cols_count:>5} "
            f"{users:>8} {missing_pct:>9} {numeric_count:>8}"
        )

    lines.append("")
    return lines


# ── main ─────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDA across all supported health-intervention datasets."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for caching downloaded datasets (default: data).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Starting EDA — loading all datasets (data_dir=%s)", args.data_dir)

    results = load_all(data_dir=args.data_dir)

    # Collect terminal-friendly output
    output_lines: list[str] = []
    output_lines.append("")
    output_lines.append(f"{'╔':═^78}")
    output_lines.append(
        f"  Exploratory Data Analysis — {datetime.now():%Y-%m-%d %H:%M}"
    )
    output_lines.append(f"{'╚':═^78}")

    for name, df in results.items():
        if df is not None:
            describe_dataset(name, df, output_lines)
        else:
            output_lines.append("")
            output_lines.append(_section(name, 78))
            output_lines.append("")
            output_lines.append(
                "  ** Dataset not loaded (missing credentials, "
                "network error, or skipped) **"
            )
            output_lines.append("")

    output_lines.extend(summary_table(results))

    # Print to terminal
    for line in output_lines:
        print(line)

    # Save to file
    report_path = os.path.join(args.data_dir, "eda_report.txt")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(output_lines))
    logger.info("EDA report saved to %s", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
