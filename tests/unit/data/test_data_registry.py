from __future__ import annotations

from rl_health_interventions.data import REGISTRY, make
from rl_health_interventions.data._base import DataConfig


def test_registry_populated() -> None:
    assert "SyntheticDataGenerator" in REGISTRY
    assert "FeaturePipeline" in REGISTRY


def test_make_synthetic() -> None:
    gen = make("SyntheticDataGenerator")
    assert gen is not None


def test_make_feature_pipeline() -> None:
    pipe = make("FeaturePipeline")
    assert pipe is not None


def test_make_unknown_raises_keyerror() -> None:
    try:
        make("NonExistent")
        assert False, "Expected KeyError"
    except KeyError as e:
        assert "NonExistent" in str(e)


def test_data_config_resolved_path_with_base_path() -> None:
    config = DataConfig(
        file_path="sub/data.csv",
        file_format="csv",
        column_mapping={"a": "b"},
        base_path="/base",
    )
    assert config.resolved_path() == "/base/sub/data.csv"


def test_data_config_resolved_path_with_env_var(monkeypatch) -> None:
    monkeypatch.setenv("RL_HEALTH_DATA_PATH", "/env/path")
    config = DataConfig(
        file_path="data.csv",
        file_format="csv",
        column_mapping={},
    )
    assert config.resolved_path() == "/env/path/data.csv"


def test_data_config_resolved_path_fail_fast() -> None:
    config = DataConfig(
        file_path="data.csv",
        file_format="csv",
        column_mapping={},
    )
    import pytest

    with pytest.raises(FileNotFoundError):
        config.resolved_path()


def test_data_config_base_path_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("RL_HEALTH_DATA_PATH", "/env/path")
    config = DataConfig(
        file_path="data.csv",
        file_format="csv",
        column_mapping={},
        base_path="/base",
    )
    assert config.resolved_path() == "/base/data.csv"
