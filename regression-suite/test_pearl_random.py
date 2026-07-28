"""Regression tests for pearl_random experiments.

Re-runs the pearl_random config at fixed seeds and compares against
the golden JSON fixture stored in docs/experimental_phases/pearl_random/results/.

Tolerance: +/-0.1% relative per metric.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNNER = (
    _REPO_ROOT / "docs" / "experimental_phases" / "pearl_random" / "run_experiments.py"
)
_RESULTS_DIR = _REPO_ROOT / "docs" / "experimental_phases" / "pearl_random" / "results"
_REL_TOLERANCE = 0.001

_METRICS = ["total_reward", "total_std", "per_step", "last50"]

_CONFIG_NAME = "pearl_random"


@pytest.fixture(scope="module")
def pearl_random_results() -> dict[str, dict]:
    """Run the pearl_random benchmark once and return results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        result = subprocess.run(
            [
                sys.executable,
                str(_RUNNER),
                "--seeds",
                "50",
                "--output",
                str(tmpdir_path),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"Runner failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

        json_file = tmpdir_path / f"{_CONFIG_NAME}.json"
        assert json_file.exists(), f"Expected output: {json_file}"
        with json_file.open(encoding="utf-8") as f:
            data = json.load(f)
        return data["agents"]


@pytest.mark.timeout(30, func_only=True)
def test_pearl_random_regression(pearl_random_results: dict[str, dict]) -> None:
    """Compare live results against the golden fixture."""
    fixture_path = _RESULTS_DIR / f"{_CONFIG_NAME}.json"
    assert fixture_path.exists(), (
        f"Golden fixture missing: {fixture_path}\n"
        f"Generate with: python {_RUNNER} "
        f"--output {_RESULTS_DIR} --json --confirm-overwrite"
    )

    with fixture_path.open(encoding="utf-8") as f:
        fixture = json.load(f)

    config_path = (
        _REPO_ROOT
        / "docs"
        / "experimental_phases"
        / "pearl_random"
        / "configs"
        / f"{_CONFIG_NAME}.yaml"
    )
    with config_path.open(encoding="utf-8") as f:
        config_seed = (yaml.safe_load(f) or {}).get("seed", 42)
    assert fixture["seed"] == config_seed, (
        f"Fixture seed ({fixture['seed']}) != config seed ({config_seed}). "
        f"Re-baseline with: python {_RUNNER}"
        f" --output {_RESULTS_DIR} --json --confirm-overwrite"
    )

    assert fixture["seeds"] == 50, (
        f"Fixture seeds ({fixture['seeds']}) != 50. "
        f"Re-baseline with: python {_RUNNER}"
        f" --seeds 50 --output {_RESULTS_DIR} --json --confirm-overwrite"
    )

    golden_agents = fixture["agents"]
    live_agents = pearl_random_results

    missing_from_live = set(golden_agents) - set(live_agents)
    extra_from_live = set(live_agents) - set(golden_agents)
    assert not missing_from_live, (
        f"Agents missing from live results: {missing_from_live}"
    )
    assert not extra_from_live, f"Extra agents in live results: {extra_from_live}"

    for agent_label, golden_metrics in golden_agents.items():
        live_metrics = live_agents[agent_label]
        for metric in _METRICS:
            golden_val = golden_metrics[metric]
            live_val = live_metrics[metric]
            assert live_val == pytest.approx(
                golden_val, rel=_REL_TOLERANCE, abs=1e-5
            ), (
                f"{_CONFIG_NAME} / {agent_label} / {metric}: "
                f"live={live_val:.6f} vs golden={golden_val:.6f}"
            )
