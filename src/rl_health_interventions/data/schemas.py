"""Schema definitions for wearable health datasets."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import yaml


class SemanticField(enum.Enum):
    """Semantic fields that the MDP environment expects."""

    STEPS = "steps"
    HEART_RATE = "heart_rate"
    SLEEP = "sleep"
    TIME_OF_DAY = "time_of_day"
    ACCELEROMETER = "accelerometer"


@dataclass
class FieldSchema:
    """Schema for a single data field."""

    column: str
    dtype: str
    unit: str | None = None
    description: str | None = None


@dataclass
class DatasetSchema:
    """Complete schema for a wearable health dataset."""

    name: str
    description: str
    source: str
    access: str
    schema_fields: dict[str, FieldSchema]
    semantic_mapping: dict[str, str | None]
    file_format: str = "csv"
    encoding: str = "utf-8"
    notes: str | None = None

    def get_semantic_field(self, field_name: str) -> str | None:
        """Get the column name for a semantic field, or None if not available."""
        return self.semantic_mapping.get(field_name)

    def get_column_dtype(self, column_name: str) -> str | None:
        """Get the dtype for a column name."""
        for field_schema in self.schema_fields.values():
            if field_schema.column == column_name:
                return field_schema.dtype
        return None

    def required_columns(self) -> list[str]:
        """Return columns that map to semantic fields (required for MDP)."""
        required = []
        for semantic_col in self.semantic_mapping.values():
            if semantic_col is not None:
                required.append(semantic_col)
        return required


def _parse_field_schema(field_data: dict) -> FieldSchema:
    """Parse a field schema from YAML dict."""
    return FieldSchema(
        column=field_data["column"],
        dtype=field_data.get("dtype", "str"),
        unit=field_data.get("unit"),
        description=field_data.get("description"),
    )


def load_schema_from_yaml(path: str | Path) -> DatasetSchema:
    """Load a dataset schema from a YAML config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed DatasetSchema.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        KeyError: If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Schema config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    dataset_config = config["dataset"]
    schema_config = config["schema"]
    semantic_config = config.get("semantic_mapping", {})

    schema_fields = {}
    for field_name, field_data in schema_config.items():
        schema_fields[field_name] = _parse_field_schema(field_data)

    return DatasetSchema(
        name=dataset_config["name"],
        description=dataset_config["description"],
        source=dataset_config["source"],
        access=dataset_config["access"],
        schema_fields=schema_fields,
        semantic_mapping=semantic_config,
        file_format=config.get("file_format", "csv"),
        encoding=config.get("encoding", "utf-8"),
        notes=config.get("notes"),
    )


def validate_schema(schema: DatasetSchema) -> list[str]:
    """Validate a dataset schema and return any errors.

    Args:
        schema: The schema to validate.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    if not schema.name:
        errors.append("Schema name is required")

    if not schema.schema_fields:
        errors.append("Schema must define at least one field")

    if not schema.semantic_mapping:
        errors.append("Schema must define semantic_mapping")

    # Check that semantic mapping references valid columns
    for semantic_field, column_name in schema.semantic_mapping.items():
        if column_name is not None:
            column_names = [f.column for f in schema.schema_fields.values()]
            if column_name not in column_names:
                errors.append(
                    f"Semantic field '{semantic_field}' maps to column "
                    f"'{column_name}' which is not in schema_fields"
                )

    # Check that STEPS is mapped (required for MDP)
    if schema.semantic_mapping.get("steps") is None:
        errors.append("Semantic field 'steps' must be mapped (required for MDP)")

    return errors
