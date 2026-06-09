"""Tests for config-driven data loading."""

from pathlib import Path

import pandas as pd
import pytest

from rl_health_interventions.data.loader import load_dataset
from rl_health_interventions.data.schemas import load_schema_from_yaml

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs" / "datasets"


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV file for testing."""
    df = pd.DataFrame(
        {
            "person_id": ["P001", "P001", "P002", "P002"],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]
            ),
            "steps": [8500, 12000, 6000, 9500],
            "resting_heart_rate": [68.0, 72.0, 55.0, 60.0],
            "heart_rate_minutes": [1440.0, 1440.0, 1440.0, 1440.0],
            "sleep_minutes": [420.0, 390.0, 480.0, 450.0],
            "sleep_efficiency": [0.92, 0.88, 0.95, 0.91],
            "calories": [2200.0, 2500.0, 1800.0, 2100.0],
            "distance_meters": [6500.0, 9200.0, 4800.0, 7300.0],
            "floors_climbed": [8, 12, 4, 7],
        }
    )
    csv_path = tmp_path / "sample_fitbit.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_config(tmp_path: Path, sample_csv: Path) -> Path:
    """Create a minimal test config."""
    config = {
        "dataset": {
            "name": "test_fitbit",
            "description": "Test dataset",
            "source": "test",
            "access": "open",
        },
        "schema": {
            "participant_id": {"column": "person_id", "dtype": "str"},
            "date": {"column": "date", "dtype": "datetime"},
            "steps": {"column": "steps", "dtype": "int"},
            "heart_rate_avg": {
                "column": "resting_heart_rate",
                "dtype": "float",
            },
            "sleep_minutes": {"column": "sleep_minutes", "dtype": "float"},
        },
        "semantic_mapping": {
            "steps": "steps",
            "heart_rate": "resting_heart_rate",
            "sleep": "sleep_minutes",
            "time_of_day": "date",
        },
        "file_format": "csv",
    }
    import yaml

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


class TestLoadDataset:
    """Tests for load_dataset."""

    def test_load_from_csv(self, sample_config: Path, tmp_path: Path) -> None:
        """Test loading a CSV dataset via config."""
        # The CSV is in tmp_path alongside the config
        csv_path = tmp_path / "sample_fitbit.csv"
        df = pd.DataFrame(
            {
                "person_id": ["P001", "P002"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "steps": [8500, 12000],
                "resting_heart_rate": [68.0, 72.0],
                "sleep_minutes": [420.0, 390.0],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_dataset(sample_config, data_dir=tmp_path)
        assert len(result) == 2
        assert "steps" in result.columns
        assert "heart_rate" in result.columns  # renamed via semantic mapping
        assert "sleep" in result.columns

    def test_load_missing_config(self) -> None:
        """Test that missing config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/config.yaml")

    def test_load_missing_data_dir(self, sample_config: Path) -> None:
        """Test that missing data directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset(sample_config, data_dir="/nonexistent/dir")

    def test_column_mapping(self, sample_config: Path, tmp_path: Path) -> None:
        """Test that columns are renamed to semantic names."""
        csv_path = tmp_path / "sample_fitbit.csv"
        df = pd.DataFrame(
            {
                "person_id": ["P001"],
                "date": pd.to_datetime(["2024-01-01"]),
                "steps": [8500],
                "resting_heart_rate": [68.0],
                "sleep_minutes": [420.0],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_dataset(sample_config, data_dir=tmp_path)
        # Original column names should be replaced with semantic names
        assert "resting_heart_rate" not in result.columns
        assert "heart_rate" in result.columns
        assert "sleep_minutes" not in result.columns
        assert "sleep" in result.columns


class TestLoadSchemaIntegration:
    """Integration tests for schema loading."""

    def test_all_of_us_schema_loads(self) -> None:
        schema = load_schema_from_yaml(CONFIGS_DIR / "all_of_us_fitbit.yaml")
        assert schema.file_format == "csv"
        assert len(schema.schema_fields) >= 5

    def test_uk_biobank_schema_loads(self) -> None:
        schema = load_schema_from_yaml(CONFIGS_DIR / "uk_biobank_accelerometer.yaml")
        assert schema.file_format == "csv"
        assert len(schema.schema_fields) >= 10
