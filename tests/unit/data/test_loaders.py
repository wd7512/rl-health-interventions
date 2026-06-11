from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
import requests

from rl_health_interventions.data.loaders import (
    _check_kaggle_auth,
    _check_physionet_auth,
    load_all,
    load_bidsleep,
    load_extrasensory,
    load_harth,
    load_mhealth,
    load_pmdata,
    load_synthetic,
    load_wesad,
    load_wisdm,
)


def test_check_kaggle_auth_no_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = _check_kaggle_auth()
    assert result is False


def test_check_kaggle_auth_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fake-token")
    result = _check_kaggle_auth()
    assert result is True


def test_check_kaggle_auth_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").touch()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = _check_kaggle_auth()
    assert result is True


def test_check_physionet_auth_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHYSIONET_USERNAME", raising=False)
    monkeypatch.delenv("PHYSIONET_PASSWORD", raising=False)
    monkeypatch.delenv("WFDB_USERNAME", raising=False)
    monkeypatch.delenv("WFDB_PASSWORD", raising=False)
    result = _check_physionet_auth()
    assert result is False


def test_check_physionet_auth_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHYSIONET_USERNAME", "user")
    monkeypatch.setenv("PHYSIONET_PASSWORD", "pass")
    result = _check_physionet_auth()
    assert result is True


def test_check_physionet_auth_wfdb_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WFDB_USERNAME", "user")
    monkeypatch.setenv("WFDB_PASSWORD", "pass")
    result = _check_physionet_auth()
    assert result is True


def test_load_synthetic_returns_dataframe() -> None:
    df = load_synthetic(n_users=10, n_timesteps=5)
    assert df is not None
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (50, 3)
    assert "user_id" in df.columns
    assert "timestep" in df.columns
    assert "steps" in df.columns


def test_load_all_returns_dict_with_all_datasets() -> None:
    results = load_all(data_dir="/tmp/nonexistent_data_dir_for_test")
    assert isinstance(results, dict)
    assert len(results) == 12
    expected = {
        "pmdata",
        "harth",
        "bidsleep",
        "scientisst_move",
        "dreamt",
        "wesad",
        "wisdm",
        "mhealth",
        "extrasensory",
        "fitbit_tracker",
        "4tu_step_goals",
        "synthetic",
    }
    assert set(results.keys()) == expected


def test_load_pmdata_returns_none_without_kaggle_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rl_health_interventions.data.loaders._check_kaggle_auth", lambda: False
    )
    result = load_pmdata(data_dir="/tmp/fake")
    assert result is None


def test_load_wesad_returns_none_without_kaggle_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rl_health_interventions.data.loaders._check_kaggle_auth", lambda: False
    )
    result = load_wesad(data_dir="/tmp/fake")
    assert result is None


def test_load_bidsleep_returns_none_without_physionet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rl_health_interventions.data.loaders._physionet_download",
        lambda *a, **kw: None,
    )
    result = load_bidsleep(data_dir="/tmp/fake")
    assert result is None


def test_load_extrasensory_returns_none_on_download_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rl_health_interventions.data.loaders.requests.head",
        Mock(side_effect=requests.ConnectionError("no network")),
    )
    monkeypatch.setattr(
        "rl_health_interventions.data.loaders._download_file",
        Mock(side_effect=Exception("no network")),
    )
    result = load_extrasensory(data_dir="/tmp/fake")
    assert result is None


def test_load_wisdm_returns_none_on_download_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rl_health_interventions.data.loaders._download_file",
        Mock(side_effect=Exception("no network")),
    )
    result = load_wisdm(data_dir="/tmp/fake_wisdm_test")
    assert result is None


def test_load_mhealth_returns_none_without_uci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ucimlrepo.fetch_ucirepo",
        Mock(side_effect=Exception("no UCI")),
    )
    result = load_mhealth(data_dir="/tmp/fake")
    assert result is None


def test_load_harth_returns_none_without_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "datasets.load_dataset",
        Mock(side_effect=Exception("no HF")),
    )
    result = load_harth(data_dir="/tmp/fake")
    assert result is None


WISDM_RAW = """33,Jogging,49105962326000,-0.6946377,12.680544,0.50395286;
33,Jogging,49105962327000,0.12774,12.435532,0.56133586;
33,Jogging,49105962328000,-0.5168617,12.250055,0.48974788;
"""


def test_wisdm_semicolon_parsing(tmp_path: Path) -> None:
    data_dir = tmp_path / "wisdm"
    data_dir.mkdir()
    raw_txt = data_dir / "WISDM_ar_v1.1_raw.txt"
    raw_txt.write_text(WISDM_RAW, encoding="utf-8")
    marker = data_dir / ".wisdm_done"
    marker.touch()

    result = load_wisdm(data_dir=str(tmp_path))
    assert result is not None
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (3, 6)
    assert list(result.columns) == [
        "user_id",
        "activity",
        "timestamp",
        "accelerometer_x",
        "accelerometer_y",
        "accelerometer_z",
    ]
    assert result["user_id"].to_list() == ["33", "33", "33"]
    assert result["activity"].to_list() == ["Jogging", "Jogging", "Jogging"]
    assert result["accelerometer_x"].to_list() == pytest.approx(
        [-0.6946377, 0.12774, -0.5168617]
    )
