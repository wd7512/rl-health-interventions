from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ASSETS = Path(__file__).parent.parent / "assets"


def test_mvp_end_to_end(tmp_path: Path) -> None:
    config_path = ASSETS / "test_integration_config.yaml"
    output_path = tmp_path / "results.csv"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl_health_interventions",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"CLI failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    assert output_path.exists()

    df = pd.read_csv(output_path)
    assert len(df) == 5  # 1 day x 5 steps/day
    # Expected columns include factor names instead of "state"
    assert set(df.columns) >= {
        "step",
        "day",
        "step_of_day",
        "activity_level",
        "action",
        "reward",
    }
    assert "state" not in df.columns
    assert df["reward"].isin([0.0, 1.0]).all()


def test_epsilon_greedy_baseline_also_works(tmp_path: Path) -> None:
    config_path = ASSETS / "test_integration_config.yaml"
    output_path = tmp_path / "results_eg.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl_health_interventions",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--agent",
            "epsilon_greedy",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == 5


def test_seed_reproducibility(tmp_path: Path) -> None:
    config_path = ASSETS / "test_integration_config.yaml"
    out1 = tmp_path / "run1.csv"
    out2 = tmp_path / "run2.csv"
    for out in [out1, out2]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rl_health_interventions",
                "--config",
                str(config_path),
                "--output",
                str(out),
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
    df1 = pd.read_csv(out1)
    df2 = pd.read_csv(out2)
    pd.testing.assert_frame_equal(df1, df2)
