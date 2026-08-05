"""Unit tests for the PEARL Constitution gap-map fixes (Phase 3).

Covers the four code-level fixes from ``docs/research/constitution-gaps.md``:

- Gap 1: T2.3 asserts RL > max(Fixed, Random, Control) with no chain
  between the three non-RL arms.
- Gap 2: T2.2 derives its band from ``load_reference()`` 1-month deltas
  with a 150-450 fallback floor.
- Gap 4: the 12-action config never inherits a persona table_dir.
- Gap 5: T4.2 reports a documented WARNING skip, not a silent PASS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.pearl_constitution.run_distribution_check import (
    check_t2_2_effect_size_magnitude,
    check_t2_3_effect_size_ordering,
)
from scripts.pearl_constitution.run_stress_tests import check_t4_2_persona_collapse
from scripts.pearl_constitution.utils import load_constitution_config

_REF = {
    "effect_sizes": {
        "rl_vs_control_1mo": {"delta": 296, "p": 0.0002},
        "rl_vs_random_1mo": {"delta": 218, "p": 0.005},
        "rl_vs_fixed_1mo": {"delta": 238, "p": 0.002},
    }
}


def _daily_steps(means: dict[str, float], n_seeds: int = 30) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    out: dict[str, np.ndarray] = {}
    for arm, mean in means.items():
        out[arm] = rng.normal(mean, 50.0, size=(n_seeds, 60))
    return out


# ── Gap 1: T2.3 ordering ──────────────────────────────────────────────────────


def test_t2_3_passes_when_rl_dominates_no_chain_between_nonrl() -> None:
    """RL > max(Fixed, Random, Control) passes even when Fixed < Random."""
    daily = _daily_steps(
        {
            "control": 5600.0,
            "random": 5750.0,
            "fixed": 5700.0,
            "rl": 5900.0,
        }
    )
    result = check_t2_3_effect_size_ordering(daily)
    assert result["passed"] is True, result


def test_t2_3_fails_when_rl_below_a_nonrl_arm() -> None:
    daily = _daily_steps(
        {
            "control": 5600.0,
            "random": 5750.0,
            "fixed": 5850.0,
            "rl": 5800.0,
        }
    )
    result = check_t2_3_effect_size_ordering(daily)
    assert result["passed"] is False, result


def test_t2_3_missing_arm_fails() -> None:
    daily = _daily_steps(
        {
            "control": 5600.0,
            "random": 5750.0,
            "fixed": 5700.0,
        }
    )
    result = check_t2_3_effect_size_ordering(daily)
    assert result["passed"] is False
    assert "Missing arm" in result["detail"]


# ── Gap 2: T2.2 derived band ──────────────────────────────────────────────────


def test_t2_2_delta_inside_reference_band_passes() -> None:
    """A 1-month lift within the reference-derived [218, 296] band passes."""
    daily = _daily_steps({"control": 5600.0, "rl": 5850.0})  # Δ=250
    result = check_t2_2_effect_size_magnitude(daily, _REF)
    assert result["passed"] is True, result
    assert "reference 1-month deltas" in result["detail"], result


def test_t2_2_delta_outside_reference_band_fails() -> None:
    """A lift far above the reference-derived band fails."""
    daily = _daily_steps({"control": 5600.0, "rl": 6200.0})  # Δ=600
    result = check_t2_2_effect_size_magnitude(daily, _REF)
    assert result["passed"] is False, result


def test_t2_2_falls_back_to_wide_floor_without_effect_sizes() -> None:
    """Without reference 1-month deltas the 150-450 floor is used."""
    ref_no_deltas: dict = {"effect_sizes": {}}
    daily = _daily_steps({"control": 5600.0, "rl": 5950.0})  # Δ=350
    result = check_t2_2_effect_size_magnitude(daily, ref_no_deltas)
    assert result["passed"] is True, result
    assert "fallback floor" in result["detail"], result


def test_t2_2_excludes_2month_and_sustained_deltas() -> None:
    """Bands use only 1-month deltas; 2-month/sustained are ignored."""
    ref = {
        "effect_sizes": {
            "rl_vs_control_1mo": {"delta": 296},
            "rl_vs_control_2mo": {"delta": 210},
            "gee_sustained": {"delta": 208},
        }
    }
    daily = _daily_steps({"control": 5600.0, "rl": 5750.0})  # Δ=150
    result = check_t2_2_effect_size_magnitude(daily, ref)
    assert result["passed"] is False, result  # 150 < 218


# ── Gap 4: 12-action persona guard ────────────────────────────────────────────


def test_12action_config_never_overrides_persona_table_dir() -> None:
    """The 12-action config keeps its own table_dir for any persona."""
    config = load_constitution_config(
        "goal_driven", "config/pearl_constitution_12action.yaml"
    )
    assert config.steps_per_day == 1
    assert config.transition_model.table_dir is not None
    assert "pearl_12action" in config.transition_model.table_dir
    assert "persona" not in config.transition_model.table_dir


def test_4action_config_does_override_persona_table_dir() -> None:
    """The original 4-action config still applies persona table dirs."""
    config = load_constitution_config("goal_driven", "config/pearl_constitution.yaml")
    assert config.steps_per_day != 1
    assert "goal_driven" in config.transition_model.table_dir


# ── Gap 5: T4.2 documented skip ──────────────────────────────────────────────


def test_t4_2_reports_documented_skip_by_default() -> None:
    """T4.2 skips with an explicit limitation note, not a silent PASS."""
    result = check_t4_2_persona_collapse({})
    assert result["passed"] is True
    assert "SKIPPED" in result["detail"]
    assert "documented limitation" in result["detail"].lower()


def test_t4_2_runs_anova_when_available(monkeypatch) -> None:
    """With PEARL_T4_2_AVAILABLE=1 and 2+ arms, the ANOVA path runs."""
    daily = _daily_steps(
        {
            "control": 5600.0,
            "random": 5750.0,
            "fixed": 5700.0,
            "rl": 5900.0,
        }
    )
    monkeypatch.setenv("PEARL_T4_2_AVAILABLE", "1")
    result = check_t4_2_persona_collapse(daily)
    assert result["detail"].startswith("F=")
    assert "SKIPPED" not in result["detail"]


def test_t4_2_anova_requires_two_arms(monkeypatch) -> None:
    """With fewer than two non-empty arms, T4.2 falls back to the skip path."""
    daily = {"control": np.empty((0, 60))}
    monkeypatch.setenv("PEARL_T4_2_AVAILABLE", "1")
    result = check_t4_2_persona_collapse(daily)
    assert "SKIPPED" in result["detail"]
