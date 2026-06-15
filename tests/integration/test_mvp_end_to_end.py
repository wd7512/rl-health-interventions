from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_mvp_end_to_end(tmp_path: Path) -> None:
    """Run the MVP config through the CLI, verify output."""
    config_path = Path(__file__).parent.parent.parent / "config" / "mvp.yaml"
    output_path = tmp_path / "results.csv"

    result = subprocess.run(
        [sys.executable, "-m", "rl_health_interventions", "--config", str(config_path), "--output", str(output_path), "--agent", "thompson_sampling"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert output_path.exists()

    df = pd.read_csv(output_path)
    assert len(df) == 450  # 90 days x 5 steps/day
    assert set(df.columns) >= {"step", "day", "step_of_day", "time_of_day", "state", "action", "reward"}
    # Verify reward is 0 or 1
    assert df["reward"].isin([0.0, 1.0]).all()


def test_epsilon_greedy_baseline_also_works(tmp_path: Path) -> None:
    """Same as above but with epsilon-greedy agent."""
    config_path = Path(__file__).parent.parent.parent / "config" / "mvp.yaml"
    output_path = tmp_path / "results_eg.csv"
    result = subprocess.run(
        [sys.executable, "-m", "rl_health_interventions", "--config", str(config_path), "--output", str(output_path), "--agent", "epsilon_greedy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == 450
