from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_runs_with_config_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
activity_levels: [sedentary, active]
actions: [send, don_t_send]
time_of_day: [morning, midday, afternoon, evening, night]
steps_per_day: 2
episode_days: 1
transition:
  sedentary:
    send:
      active: 0.5
      sedentary: 0.5
    don_t_send:
      active: 0.5
      sedentary: 0.5
  active:
    send:
      active: 0.5
      sedentary: 0.5
    don_t_send:
      active: 0.5
      sedentary: 0.5
masks:
  morning: {sedentary: 0.0, active: 0.0}
  midday: {sedentary: 0.0, active: 0.0}
  afternoon: {sedentary: 0.0, active: 0.0}
  evening: {sedentary: 0.0, active: 0.0}
  night: {sedentary: 0.0, active: 0.0}
""")
    output_path = tmp_path / "results.csv"
    result = subprocess.run(
        [sys.executable, "-m", "rl_health_interventions", "--config", str(config_path), "--output", str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert output_path.exists()
