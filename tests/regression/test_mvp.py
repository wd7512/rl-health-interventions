"""Regression tests for MVP experiments.

Re-runs all MVP configs at fixed seeds and compares against golden JSON fixtures
stored in docs/experimental_phases/mvp/results/.

Tolerance: ±0.1% relative per metric.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RUNNER = _REPO_ROOT / "docs" / "experimental_phases" / "mvp" / "run_experiments.py"
_RESULTS_DIR = _REPO_ROOT / "docs" / "experimental_phases" / "mvp" / "results"
_REL_TOLERANCE = 0.001  # 0.1% relative tolerance

_METRICS = ["total_reward", "total_std", "per_step", "last50"]


@pytest.fixture(scope="session")
def mvp_results() -> dict[str, dict]:
    """Run the MVP benchmark once and return all config results as structured data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        result = subprocess.run(
            [
                sys.executable,
                str(_RUNNER),
                "--all",
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

        results: dict[str, dict] = {}
        for json_file in sorted(tmpdir_path.glob("*.json")):
            config_name = json_file.stem
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
            results[config_name] = data["agents"]
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
        f"Generate with: python {_RUNNER} --config {config_name} "
        f"--output {_RESULTS_DIR} --json --confirm-overwrite"
    )

    with fixture_path.open(encoding="utf-8") as f:
        fixture = json.load(f)

    # Guard against seed drift: fixture seed must match config seed
    config_path = (
        _REPO_ROOT
        / "docs"
        / "experimental_phases"
        / "mvp"
        / "configs"
        / f"{config_name}.yaml"
    )
    with config_path.open(encoding="utf-8") as f:
        config_seed = (yaml.safe_load(f) or {}).get("seed", 42)
    assert fixture["seed"] == config_seed, (
        f"Fixture seed ({fixture['seed']}) != config seed ({config_seed}). "
        f"Re-baseline with: python {_RUNNER} --config {config_name}"
        f" --output {_RESULTS_DIR} --json --confirm-overwrite"
    )

    # Guard against seed count drift
    assert fixture["seeds"] == 50, (
        f"Fixture seeds ({fixture['seeds']}) != 50. "
        f"Re-baseline with: python {_RUNNER} --config {config_name}"
        f" --seeds 50 --output {_RESULTS_DIR} --json --confirm-overwrite"
    )

    golden_agents = fixture["agents"]
    live_agents = mvp_results.get(config_name, {})

    missing_from_live = set(golden_agents) - set(live_agents)
    extra_from_live = set(live_agents) - set(golden_agents)
    assert not missing_from_live, (
        f"Agents missing from live results for {config_name}: {missing_from_live}"
    )
    assert not extra_from_live, (
        f"Extra agents in live results for {config_name}: {extra_from_live}"
    )

    for agent_label, golden_metrics in golden_agents.items():
        live_metrics = live_agents[agent_label]
        for metric in _METRICS:
            golden_val = golden_metrics[metric]
            live_val = live_metrics[metric]
            assert live_val == pytest.approx(
                golden_val, rel=_REL_TOLERANCE, abs=1e-5
            ), (
                f"{config_name} / {agent_label} / {metric}: "
                f"live={live_val:.6f} vs golden={golden_val:.6f}"
            )
