"""Regression test: total rewards must match the committed fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_health_interventions.experiment import run_experiment

FIXTURE = Path("tests/fixtures/mvp_expected_rewards.json")


def test_mvp_total_rewards_unchanged() -> None:
    if not FIXTURE.exists():
        pytest.skip(
            "Regression fixture not found; run scripts/generate_regression_fixture.py"
        )
    results = run_experiment("config/rule_based.yaml")
    expected = json.loads(FIXTURE.read_text())
    for agent_type, expected_reward in expected.items():
        assert results[agent_type] == pytest.approx(expected_reward, abs=1e-6), (
            f"Agent '{agent_type}' reward changed: "
            f"got {results[agent_type]}, expected {expected_reward}"
        )
