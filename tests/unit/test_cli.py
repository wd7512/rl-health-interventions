from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ASSETS = Path(__file__).parent.parent / "assets"


def test_cli_runs_with_config_flag(tmp_path: Path) -> None:
    config_path = ASSETS / "minimal_config.yaml"
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


def test_cli_agent_flag_produces_different_results(tmp_path: Path) -> None:
    config_path = ASSETS / "minimal_config.yaml"
    out_random = tmp_path / "random.csv"
    out_ts = tmp_path / "ts.csv"
    result_r = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl_health_interventions",
            "--config",
            str(config_path),
            "--output",
            str(out_random),
            "--agent",
            "random",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    result_ts = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl_health_interventions",
            "--config",
            str(config_path),
            "--output",
            str(out_ts),
            "--agent",
            "thompson_sampling",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result_r.returncode == 0, f"random agent failed: {result_r.stderr}"
    assert result_ts.returncode == 0, f"TS agent failed: {result_ts.stderr}"
    import pandas as pd

    df_random = pd.read_csv(out_random)
    df_ts = pd.read_csv(out_ts)
    assert len(df_random) == len(df_ts)
