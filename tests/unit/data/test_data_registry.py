from __future__ import annotations

import pytest

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
    with pytest.raises(KeyError, match="NonExistent"):
        make("NonExistent")


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


def test_dataset_validate_happy_path() -> None:
    import numpy as np

    from rl_health_interventions.data.dataset import Dataset

    ds = Dataset(
        user_ids=np.arange(3, dtype=np.int64),
        timestamps=np.empty((3, 5), dtype=np.int64),
        features={"steps": np.ones((3, 5), dtype=np.int64)},
        metadata={"n_users": 3, "n_timesteps": 5},
    )
    ds.validate()  # should not raise


def test_dataset_validate_mismatched_timestamps() -> None:
    import numpy as np
    import pytest

    from rl_health_interventions.data.dataset import Dataset

    ds = Dataset(
        user_ids=np.arange(3, dtype=np.int64),
        timestamps=np.empty((2, 5), dtype=np.int64),  # wrong
        features={"steps": np.ones((3, 5), dtype=np.int64)},
        metadata={"n_users": 3, "n_timesteps": 5},
    )
    with pytest.raises(ValueError, match="timestamps"):
        ds.validate()


def test_dataset_validate_mismatched_features() -> None:
    import numpy as np
    import pytest

    from rl_health_interventions.data.dataset import Dataset

    ds = Dataset(
        user_ids=np.arange(3, dtype=np.int64),
        timestamps=np.empty((3, 5), dtype=np.int64),
        features={"steps": np.ones((2, 5), dtype=np.int64)},  # wrong
        metadata={"n_users": 3, "n_timesteps": 5},
    )
    with pytest.raises(ValueError, match="steps"):
        ds.validate()


def test_dataset_validate_empty_arrays() -> None:
    import numpy as np

    from rl_health_interventions.data.dataset import Dataset

    ds = Dataset(
        user_ids=np.array([], dtype=np.int64),
        timestamps=np.empty((0, 5), dtype=np.int64),
        features={"steps": np.empty((0, 5), dtype=np.int64)},
        metadata={"n_users": 0, "n_timesteps": 5},
    )
    ds.validate()  # n_users=0 is valid


def test_data_config_empty_file_path_rejected() -> None:
    import pytest
    from pydantic import ValidationError

    from rl_health_interventions.data._base import DataConfig

    with pytest.raises(ValidationError):
        DataConfig(
            file_path="",
            file_format="csv",
            column_mapping={},
        )


def test_data_config_invalid_file_format_rejected() -> None:
    import pytest
    from pydantic import ValidationError

    from rl_health_interventions.data._base import DataConfig

    # Build dict and validate, bypassing type checker
    from pydantic import TypeAdapter
    adapter = TypeAdapter(DataConfig)
    with pytest.raises(ValidationError):
        adapter.validate_python({
            "file_path": "data.tsv",
            "file_format": "tsv",
            "column_mapping": {},
        })


def test_data_config_base_path_empty_string_falls_through(monkeypatch) -> None:
    """Empty string base_path should be treated as unset (fall through to env var)."""
    monkeypatch.setenv("RL_HEALTH_DATA_PATH", "/env/path")
    config = DataConfig(
        file_path="data.csv",
        file_format="csv",
        column_mapping={},
        base_path="",
    )
    assert config.resolved_path() == "/env/path/data.csv"


def test_synthetic_generator_output_shapes() -> None:
    from rl_health_interventions.data.synthetic import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    ds = gen.generate(n_users=10, n_timesteps=5)
    assert ds.user_ids.shape == (10,)
    assert ds.timestamps.shape == (10, 5)
    assert "steps" in ds.features
    assert ds.features["steps"].shape == (10, 5)
    assert ds.metadata["n_users"] == 10
    assert ds.metadata["n_timesteps"] == 5


def test_synthetic_generator_seed_determinism() -> None:
    from rl_health_interventions.data.synthetic import SyntheticDataGenerator

    gen1 = SyntheticDataGenerator(seed=42)
    gen2 = SyntheticDataGenerator(seed=42)
    ds1 = gen1.generate(n_users=5, n_timesteps=3)
    ds2 = gen2.generate(n_users=5, n_timesteps=3)
    import numpy as np
    np.testing.assert_array_equal(ds1.user_ids, ds2.user_ids)
    np.testing.assert_array_equal(ds1.features["steps"], ds2.features["steps"])


def test_synthetic_generator_zero_users() -> None:
    from rl_health_interventions.data.synthetic import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    ds = gen.generate(n_users=0, n_timesteps=5)
    assert ds.user_ids.shape == (0,)
    assert ds.timestamps.shape == (0, 5)
    ds.validate()  # should not raise
