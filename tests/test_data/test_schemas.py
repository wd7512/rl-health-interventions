"""Tests for dataset schema loading and validation."""

from pathlib import Path

import pytest

from rl_health_interventions.data.schemas import (
    DatasetSchema,
    FieldSchema,
    load_schema_from_yaml,
    validate_schema,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs" / "datasets"


class TestLoadSchemaFromYaml:
    """Tests for load_schema_from_yaml."""

    def test_load_all_of_us_schema(self) -> None:
        schema = load_schema_from_yaml(CONFIGS_DIR / "all_of_us_fitbit.yaml")
        assert schema.name == "all_of_us_fitbit"
        assert schema.access == "controlled_access"
        assert "steps" in schema.schema_fields
        assert schema.semantic_mapping["steps"] == "steps"
        assert schema.semantic_mapping["heart_rate"] == "resting_heart_rate"

    def test_load_uk_biobank_schema(self) -> None:
        schema = load_schema_from_yaml(CONFIGS_DIR / "uk_biobank_accelerometer.yaml")
        assert schema.name == "uk_biobank_accelerometer"
        assert "enmo" in schema.schema_fields
        assert schema.semantic_mapping["accelerometer"] == "enmo"
        assert schema.semantic_mapping["heart_rate"] is None

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_schema_from_yaml("/nonexistent/path.yaml")

    def test_schema_fields_have_columns(self) -> None:
        schema = load_schema_from_yaml(CONFIGS_DIR / "all_of_us_fitbit.yaml")
        for field_name, field_schema in schema.schema_fields.items():
            assert isinstance(field_schema, FieldSchema)
            assert field_schema.column  # non-empty
            assert field_schema.dtype  # non-empty


class TestValidateSchema:
    """Tests for schema validation."""

    def test_valid_schema(self) -> None:
        schema = load_schema_from_yaml(CONFIGS_DIR / "all_of_us_fitbit.yaml")
        errors = validate_schema(schema)
        assert errors == []

    def test_valid_uk_biobank_schema(self) -> None:
        schema = load_schema_from_yaml(CONFIGS_DIR / "uk_biobank_accelerometer.yaml")
        errors = validate_schema(schema)
        assert errors == []

    def test_missing_steps_mapping(self) -> None:
        schema = DatasetSchema(
            name="test",
            description="test",
            source="test",
            access="test",
            schema_fields={},
            semantic_mapping={"heart_rate": "hr"},  # no steps
        )
        errors = validate_schema(schema)
        assert any("steps" in e for e in errors)

    def test_empty_name(self) -> None:
        schema = DatasetSchema(
            name="",
            description="test",
            source="test",
            access="test",
            schema_fields={"s": FieldSchema(column="s", dtype="int")},
            semantic_mapping={"steps": "s"},
        )
        errors = validate_schema(schema)
        assert any("name" in e for e in errors)


class TestDatasetSchema:
    """Tests for DatasetSchema methods."""

    def test_get_semantic_field(self) -> None:
        schema = load_schema_from_yaml(CONFIGS_DIR / "all_of_us_fitbit.yaml")
        assert schema.get_semantic_field("steps") == "steps"
        assert schema.get_semantic_field("heart_rate") == "resting_heart_rate"
        assert schema.get_semantic_field("nonexistent") is None

    def test_required_columns(self) -> None:
        schema = load_schema_from_yaml(CONFIGS_DIR / "all_of_us_fitbit.yaml")
        required = schema.required_columns()
        assert "steps" in required
        assert "resting_heart_rate" in required
        assert "sleep_minutes" in required

    def test_required_columns_uk_biobank(self) -> None:
        schema = load_schema_from_yaml(CONFIGS_DIR / "uk_biobank_accelerometer.yaml")
        required = schema.required_columns()
        assert "steps" in required
        assert "wear_time" in required
        # heart_rate is None, so not required
        assert all(col != "heart_rate" for col in required)
