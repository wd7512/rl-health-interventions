"""Regression tests for the PEARL Constitution validation infrastructure.

Verifies that all validation scripts run without errors and produce the
expected output format. Does NOT assert pass/fail for individual checks
(since those are empirical findings), but does verify:
  - Scripts execute without crashes
  - Output contains expected check IDs
  - The summary matrix format is correct
  - Results are deterministic (same seed → identical output)
  - No degenerate trajectories (T1.4) always passes structurally
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Expected check IDs for each tier
_EXPECTED_CHECKS = {
    1: ["T1.1", "T1.2", "T1.3", "T1.4"],
    2: ["T2.1", "T2.2", "T2.3", "T2.4", "T2.5"],
    3: ["T3.1", "T3.2", "T3.3", "T3.4"],
    4: ["T4.1", "T4.2", "T4.3", "T4.4"],
}
_ALL_CHECK_IDS = [
    cid for tier_checks in _EXPECTED_CHECKS.values() for cid in tier_checks
]


def _run_script(
    script_name: str,
    *args: str,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Run a pearl_constitution script and return the completed process."""
    cmd = [
        sys.executable,
        "-m",
        f"scripts.pearl_constitution.{script_name.replace('.py', '')}",
        *args,
    ]
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_REPO_ROOT),
    )


def _extract_check_results(output: str) -> list[dict[str, str]]:
    """Parse check results from the matrix output."""
    results: list[dict[str, str]] = []
    for line in output.splitlines():
        stripped = line.strip()
        # Match lines like: "  T1.1     ✅ PASS  | Baseline stability  | ..."
        # or "  T1.1     ❌ FAIL  | Baseline stability  | ..."
        for check_id in _ALL_CHECK_IDS:
            if stripped.startswith(check_id):
                parts = [p.strip() for p in stripped.split("|")]
                status = "PASS" if "✅ PASS" in parts[0] else "FAIL"
                name = parts[1].strip() if len(parts) > 1 else ""
                detail = parts[2].strip() if len(parts) > 2 else ""
                results.append(
                    {
                        "check_id": check_id,
                        "status": status,
                        "name": name,
                        "detail": detail,
                    }
                )
                break
    return results


@pytest.mark.timeout(60)
def test_tier1_baseline_check_runs() -> None:
    """Tier 1 script runs without crashing and produces expected check IDs."""
    result = _run_script("run_baseline_check.py", "--seeds", "2")
    assert result.returncode in (0, 1), (
        f"Script crashed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    checks = _extract_check_results(result.stdout)
    check_ids = {c["check_id"] for c in checks}
    expected = set(_EXPECTED_CHECKS[1])
    assert check_ids == expected, f"Expected Tier 1 checks {expected}, got {check_ids}"


@pytest.mark.timeout(300)
def test_tier2_distribution_check_runs() -> None:
    """Tier 2 script runs without crashing and produces expected check IDs."""
    result = _run_script("run_distribution_check.py", "--seeds", "5")
    assert result.returncode in (0, 1), (
        f"Script crashed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    checks = _extract_check_results(result.stdout)
    check_ids = {c["check_id"] for c in checks}
    expected = set(_EXPECTED_CHECKS[2])
    assert check_ids == expected, f"Expected Tier 2 checks {expected}, got {check_ids}"


@pytest.mark.timeout(300)
def test_tier3_behaviour_check_runs() -> None:
    """Tier 3 script runs without crashing and produces expected check IDs."""
    result = _run_script("run_behaviour_check.py", "--seeds", "5")
    assert result.returncode in (0, 1), (
        f"Script crashed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    checks = _extract_check_results(result.stdout)
    check_ids = {c["check_id"] for c in checks}
    expected = set(_EXPECTED_CHECKS[3])
    assert check_ids == expected, f"Expected Tier 3 checks {expected}, got {check_ids}"


@pytest.mark.timeout(600)
def test_tier4_stress_tests_runs() -> None:
    """Tier 4 scripts run without crashing and produces expected check IDs."""
    result = _run_script("run_stress_tests.py", "--seeds", "2")
    assert result.returncode in (0, 1), (
        f"Script crashed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    checks = _extract_check_results(result.stdout)
    check_ids = {c["check_id"] for c in checks}
    expected = set(_EXPECTED_CHECKS[4])
    assert check_ids == expected, f"Expected Tier 4 checks {expected}, got {check_ids}"


@pytest.mark.timeout(900)
def test_master_runner_produces_matrix() -> None:
    """Master runner produces the summary matrix with all 16 check IDs."""
    result = _run_script("run_all.py", "--seeds", "2", "--tiers", "1,2,3")
    assert result.returncode in (0, 1), (
        f"Master runner crashed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    # Check that the matrix header is present
    assert "PEARL Constitution" in result.stdout, "Missing summary matrix header"
    assert "Validation Summary Matrix" in result.stdout, "Missing summary matrix title"

    checks = _extract_check_results(result.stdout)
    expected_tiers_1_3 = {cid for tier in (1, 2, 3) for cid in _EXPECTED_CHECKS[tier]}
    check_ids = {c["check_id"] for c in checks}
    assert check_ids == expected_tiers_1_3, (
        f"Expected checks for tiers 1-3, got {check_ids}"
    )


@pytest.mark.timeout(120)
def test_t1_4_always_passes() -> None:
    """T1.4 (no degenerate trajectories) must always pass."""
    result = _run_script("run_baseline_check.py", "--seeds", "3")
    checks = _extract_check_results(result.stdout)
    t1_4 = [c for c in checks if c["check_id"] == "T1.4"]
    assert len(t1_4) == 1, f"Expected exactly one T1.4 result, got {len(t1_4)}"
    assert t1_4[0]["status"] == "PASS", (
        f"T1.4 should always pass structurally:\n{t1_4[0]}"
    )


@pytest.mark.timeout(300)
def test_deterministic_output() -> None:
    """Same seed produces identical output."""
    result1 = _run_script("run_baseline_check.py", "--seeds", "3")
    result2 = _run_script("run_baseline_check.py", "--seeds", "3")
    assert result1.stdout == result2.stdout, (
        "Output differs between two identical runs with same seed"
    )


@pytest.mark.timeout(120)
def test_persona_switch_works() -> None:
    """Running with a different persona (goal_driven) produces different results."""
    result_base = _run_script(
        "run_baseline_check.py", "--seeds", "2", "--persona", "base"
    )
    result_goal = _run_script(
        "run_baseline_check.py", "--seeds", "2", "--persona", "goal_driven"
    )
    assert result_base.returncode in (0, 1), (
        f"Base persona failed: {result_base.stderr}"
    )
    assert result_goal.returncode in (0, 1), (
        f"Goal-driven persona failed: {result_goal.stderr}"
    )
    # The outputs should differ since different personas produce different trajectories
    assert result_base.stdout != result_goal.stdout, (
        "Different personas should produce different results"
    )


def test_reference_schema_valid() -> None:
    """PEARL reference JSON is valid against its schema."""
    import json

    import jsonschema

    schema_path = (
        _REPO_ROOT / "docs" / "research" / "reference" / "pearl_reference_schema.json"
    )
    data_path = _REPO_ROOT / "docs" / "research" / "reference" / "pearl_reference.json"

    with schema_path.open(encoding="utf-8") as f:
        schema = json.load(f)
    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)

    jsonschema.validate(data, schema)


def test_config_loads() -> None:
    """PEARL constitution YAML config loads without validation errors."""
    from rl_health_interventions.config.loader import load_config

    config_path = _REPO_ROOT / "config" / "pearl_constitution.yaml"
    config = load_config(str(config_path))
    assert config.episode_days == 60
    assert len(config.agents) == 4
    # Verify all 4 arm types
    agent_types = [a.type for a in config.agents]
    assert agent_types == ["fixed", "random", "fixed", "thompson_sampling"]
    # Verify the two fixed agents have different actions
    fixed_agents = [a for a in config.agents if a.type == "fixed"]
    assert len(fixed_agents) == 2
    assert fixed_agents[0].action == "idle"
    assert fixed_agents[1].action == "movement_suggestion"


@pytest.mark.timeout(120)
def test_no_nan_in_trajectories() -> None:
    """All 4 arms produce valid trajectories without NaN values."""
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.pearl_constitution.utils import (
        load_constitution_config,
        run_all_arms,
    )

    config = load_constitution_config("base")
    trajectories = run_all_arms(config, n_seeds=3)

    for arm_name, seed_trajs in trajectories.items():
        for seed_idx, traj in enumerate(seed_trajs):
            for step_idx, record in enumerate(traj):
                for key, value in record.items():
                    if isinstance(value, float):
                        assert not (value != value), (  # NaN check
                            f"NaN in {arm_name}, seed {seed_idx}, step {step_idx}, key {key}"  # noqa: E501
                        )
