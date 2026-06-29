"""Regression tests for MVP experiments.

Re-runs all MVP configs at fixed seeds and compares against golden JSON fixtures
stored in docs/experimental_phases/mvp/results/.

Tolerance: ±0.1% relative per metric.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RUNNER = _REPO_ROOT / "docs" / "experimental_phases" / "mvp" / "run_experiments.py"
_RESULTS_DIR = _REPO_ROOT / "docs" / "experimental_phases" / "mvp" / "results"
_TOLERANCE = 0.001  # 0.1% relative tolerance

_METRICS = ["total_reward", "per_step", "last50"]


@pytest.fixture(scope="session")
def mvp_results() -> dict[str, dict]:
    """Run the MVP benchmark once and parse all config results."""
    result = subprocess.run(
        [sys.executable, str(_RUNNER), "--all", "--seeds", "50"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"Runner failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    return _parse_stdout(result.stdout)


def _parse_stdout(stdout: str) -> dict[str, dict]:
    """Parse the human-readable table output into structured results.

    Returns {config_name: {agent_label: {metric: value}}}.
    """
    results: dict[str, dict] = {}
    current_config: str | None = None
    lines = stdout.splitlines()

    for line in lines:
        # Detect config header
        if line.startswith("Config: "):
            # Config: .../configs/mvp.yaml
            parts = line.split("configs/")
            if len(parts) >= 2:
                current_config = parts[-1].replace(".yaml", "")
                results[current_config] = {}
            continue

        if current_config is None:
            continue

        # Skip decorative lines
        if not line.strip() or line.startswith("=") or line.startswith("----"):
            continue

        # Skip table header
        if line.strip().startswith("Agent ") and "Total" in line:
            continue

        # Parse agent line: "Ctx UCB   179.4 +- 15.4   0.3987   0.4300"
        # First try to extract leading label then parse the numbers
        stripped = line.strip()
        if not stripped:
            continue

        # Split on "+-" to isolate the label and first value
        if "+-" not in stripped:
            continue

        label_part, rest = stripped.split("+-", 1)
        label = label_part.strip()
        if not label:
            continue

        numbers_str = rest.strip().split()
        # numbers_str = [total_std, per_step, last50]
        if len(numbers_str) < 3:
            continue

        try:
            results[current_config][label] = {
                "per_step": float(numbers_str[0]),
                "last50": float(numbers_str[1]),
                "total_reward": float(numbers_str[2]),
            }
        except (ValueError, IndexError):
            # Try alternate parsing: maybe the numbers come in order total, per_step, last50
            logger.warning("Failed to parse line for %s: %s", current_config, stripped)
            continue

    return results


@pytest.mark.parametrize(
    "config_name",
    ["mvp", "mvp_masked", "mvp_extensions", "mvp_extensions_masked"],
)
def test_mvp_regression(config_name: str, mvp_results: dict[str, dict]) -> None:
    """Compare live results against golden fixtures."""
    fixture_path = _RESULTS_DIR / f"{config_name}.json"
    assert fixture_path.exists(), (
        f"Golden fixture missing: {fixture_path}\n"
        f"Generate with: python {_RUNNER} --config {config_name} --output {_RESULTS_DIR} --json --confirm-overwrite"
    )

    with fixture_path.open(encoding="utf-8") as f:
        fixture = json.load(f)

    golden_agents = fixture["agents"]
    live_agents = mvp_results.get(config_name, {})

    missing_from_live = set(golden_agents.keys()) - set(live_agents.keys())
    assert not missing_from_live, (
        f"Agents in fixture but not in live run: {missing_from_live}"
    )

    for agent_label, golden_metrics in golden_agents.items():
        live_metrics = live_agents.get(agent_label)
        assert live_metrics is not None, (
            f"Agent '{agent_label}' missing from live run for {config_name}"
        )
        for metric in _METRICS:
            golden_val = golden_metrics[metric]
            live_val = live_metrics[metric]
            rel_diff = abs(live_val - golden_val) / max(abs(golden_val), 1e-9)
            assert rel_diff <= _TOLERANCE, (
                f"{config_name} / {agent_label} / {metric}: "
                f"live={live_val:.6f} vs golden={golden_val:.6f} "
                f"(rel_diff={rel_diff:.6f} > {_TOLERANCE})"
            )
