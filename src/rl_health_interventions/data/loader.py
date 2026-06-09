"""Config-driven data loading for wearable health datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from rl_health_interventions.data.schemas import DatasetSchema, load_schema_from_yaml

logger = logging.getLogger(__name__)


def load_dataset(
    config_path: str | Path,
    data_dir: str | Path = ".",
) -> pd.DataFrame:
    """Load a dataset using a YAML config file.

    The config defines the schema (column names, dtypes) and semantic
    mapping (which columns map to MDP fields). This function loads the
    data files, validates the schema, and renames columns to semantic names.

    Args:
        config_path: Path to the YAML dataset config.
        data_dir: Directory containing the data files.

    Returns:
        DataFrame with columns named by semantic fields.

    Raises:
        FileNotFoundError: If config or data files don't exist.
        ValueError: If required columns are missing.
    """
    config_path = Path(config_path)
    data_dir = Path(data_dir)

    schema = load_schema_from_yaml(config_path)
    logger.info("Loaded schema: %s (%s)", schema.name, schema.description)

    # Find data files
    data_files = _find_data_files(data_dir, schema)
    if not data_files:
        raise FileNotFoundError(f"No {schema.file_format} files found in {data_dir}")

    logger.info("Found %d data file(s) in %s", len(data_files), data_dir)

    # Load and concatenate
    frames = []
    for f in data_files:
        df = _load_single_file(f, schema)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows from %d files", len(df), len(data_files))

    # Validate required columns
    missing = _check_required_columns(df, schema)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available: {list(df.columns)}"
        )

    # Rename to semantic names
    df = _apply_semantic_mapping(df, schema)

    return df


def _find_data_files(data_dir: Path, schema: DatasetSchema) -> list[Path]:
    """Find data files matching the schema's file format."""
    pattern = f"*.{schema.file_format}"
    files = sorted(data_dir.glob(pattern))
    return files


def _load_single_file(filepath: Path, schema: DatasetSchema) -> pd.DataFrame:
    """Load a single data file with type coercion."""
    if schema.file_format == "csv":
        df = pd.read_csv(filepath, encoding=schema.encoding)
    elif schema.file_format == "parquet":
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {schema.file_format}")

    logger.debug("Loaded %s: %d rows, %d cols", filepath.name, len(df), len(df.columns))

    # Coerce dtypes based on schema
    for field_name, field_schema in schema.schema_fields.items():
        col = field_schema.column
        if col in df.columns:
            if field_schema.dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif field_schema.dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif field_schema.dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _check_required_columns(df: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    """Check that required columns exist in the DataFrame."""
    required = schema.required_columns()
    return [col for col in required if col not in df.columns]


def _apply_semantic_mapping(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Rename columns to semantic field names."""
    rename_map = {}
    for semantic_field, column_name in schema.semantic_mapping.items():
        if column_name is not None and column_name in df.columns:
            rename_map[column_name] = semantic_field

    df = df.rename(columns=rename_map)
    logger.debug("Renamed columns: %s", rename_map)
    return df
