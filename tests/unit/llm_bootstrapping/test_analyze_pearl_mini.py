"""Unit tests for analyze_pearl_mini constitution proxy checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl_health_interventions.llm_bootstrapping.prompts.pearl import (  # noqa: E402
    ACTIONS,
)
from scripts.pearl_recalibration.analyze_pearl_mini import (  # noqa: E402
    CHECK_THRESHOLDS,
    CONTROL_ACTION,
    INTERVENTION_ACTION,
    compute_metrics,
    compute_raw_effect,
)


def _state(rsm: str, burden: str) -> dict:
    return {
        "recent_steps_mean": rsm,
        "recent_walk_pattern": "low",
        "morning_steps_ratio": "balanced",
        "day_of_week": "weekday",
        "burden": burden,
    }


def _transition(
    rsm: str,
    burden: str,
    action: str,
    *,
    p_high: float = 0.0,
    p_mod: float = 0.0,
    p_low: float = 0.0,
    walk: str = "low",
    morning_ratio: str = "balanced",
) -> dict:
    return {
        "state": _state(rsm, burden),
        "action": action,
        "next_state_probs": {
            "recent_steps_mean": {
                "low": p_low,
                "moderate": p_mod,
                "high": p_high,
            },
            "recent_walk_pattern": {walk: 1.0},
            "morning_steps_ratio": {morning_ratio: 1.0},
        },
    }


@pytest.fixture
def good_table() -> dict:  # noqa: PLR0912
    """A table with strong signals on every check.

    Sparse per-cell distributions (2 values max) so no factor value is
    present in more than 75% of cells (C6), while idle is persistent (C3)
    and interventions raise P(high) (C4).
    """
    transitions = []
    for rsm in ("low", "high"):
        for burden in ("none", "major"):
            for action in ACTIONS:
                is_intervention = action != CONTROL_ACTION
                if rsm == "low":
                    if is_intervention:
                        p_high, p_mod, p_low = 0.1, 0.4, 0.5
                    else:
                        p_high, p_mod, p_low = 0.0, 0.0, 1.0
                elif is_intervention:
                    p_high, p_mod, p_low = 1.0, 0.0, 0.0
                else:
                    p_high, p_mod, p_low = 0.6, 0.4, 0.0
                if action == CONTROL_ACTION or action.endswith("_afternoon"):
                    morning_ratio = "evening"
                elif action.endswith("_morning"):
                    morning_ratio = "morning"
                else:
                    morning_ratio = "balanced"
                transitions.append(
                    _transition(
                        rsm,
                        burden,
                        action,
                        p_high=p_high,
                        p_mod=p_mod,
                        p_low=p_low,
                        walk="low" if rsm == "low" else "high",
                        morning_ratio=morning_ratio,
                    )
                )
    return {"global_state": {}, "transitions": transitions}


def test_good_table_passes_all_checks(good_table: dict) -> None:
    metrics = compute_metrics(good_table)
    assert all(check["pass"] for check in metrics["checks"].values())


def test_persistence_fails_when_high_collapses() -> None:
    transitions = [
        _transition("low", "none", CONTROL_ACTION, p_low=1.0),
        _transition("high", "none", CONTROL_ACTION, p_mod=1.0),
        _transition("low", "none", INTERVENTION_ACTION, p_low=1.0),
        _transition("high", "none", INTERVENTION_ACTION, p_mod=1.0),
    ]
    metrics = compute_metrics({"global_state": {}, "transitions": transitions})
    assert metrics["persistence"]["high"]["mean_p_stay"] == 0.0
    assert not metrics["checks"]["C3_state_persistence"]["pass"]


def test_sensitivity_fails_when_intervention_has_no_effect() -> None:
    transitions = [
        _transition("low", "none", CONTROL_ACTION, p_low=1.0),
        _transition("low", "none", INTERVENTION_ACTION, p_low=1.0),
        _transition("high", "none", CONTROL_ACTION, p_mod=1.0),
        _transition("high", "none", INTERVENTION_ACTION, p_mod=1.0),
    ]
    metrics = compute_metrics({"global_state": {}, "transitions": transitions})
    assert metrics["checks"]["C4_action_sensitivity"]["actual"] == 0.0
    assert not metrics["checks"]["C4_action_sensitivity"]["pass"]


def test_morning_ratio_collapse_fails_variation() -> None:
    transitions = [
        _transition("low", "none", CONTROL_ACTION, p_low=1.0, morning_ratio="balanced"),
        _transition(
            "high", "none", CONTROL_ACTION, p_mod=1.0, morning_ratio="balanced"
        ),
        _transition(
            "low", "none", INTERVENTION_ACTION, p_mod=1.0, morning_ratio="balanced"
        ),
        _transition(
            "high", "none", INTERVENTION_ACTION, p_mod=1.0, morning_ratio="balanced"
        ),
    ]
    metrics = compute_metrics({"global_state": {}, "transitions": transitions})
    variation = metrics["factor_variation"]["morning_steps_ratio"]
    assert variation["dominant_share"] == 1.0
    assert not metrics["checks"]["C6_factor_variation"]["pass"]


def test_burden_monotonicity_fails_when_major_raises_steps() -> None:
    transitions = [
        _transition("high", "none", CONTROL_ACTION, p_high=0.4, p_mod=0.6),
        _transition("high", "major", CONTROL_ACTION, p_high=0.8, p_mod=0.2),
        _transition("high", "none", INTERVENTION_ACTION, p_high=0.5, p_mod=0.5),
        _transition("high", "major", INTERVENTION_ACTION, p_high=0.9, p_mod=0.1),
    ]
    metrics = compute_metrics({"global_state": {}, "transitions": transitions})
    assert not metrics["checks"]["C5_burden_monotonicity"]["pass"]
    assert metrics["monotonicity"]["high"]["burden_reduces_steps"] is False


def test_thresholds_are_sane() -> None:
    assert CHECK_THRESHOLDS["min_action_coverage"] == 1.0
    assert CHECK_THRESHOLDS["min_cell_coverage"] == 1.0
    assert 0.0 < CHECK_THRESHOLDS["min_sensitivity_frac"] <= 1.0


def _raw_record(rsm: str, action: str, morning: int, afternoon: int) -> dict:
    days = "\n".join(
        f'{{"day": {d}, "morning_steps": {morning}, "afternoon_steps": {afternoon}}}'
        for d in range(1, 8)
    )
    return {"state": _state(rsm, "none"), "action": action, "content": days}


def test_compute_raw_effect_reports_lift() -> None:
    records = [
        _raw_record("low", CONTROL_ACTION, 1500, 1500),
        _raw_record("low", INTERVENTION_ACTION, 1650, 1650),
        {"state": _state("low", "none"), "action": CONTROL_ACTION, "error": "boom"},
    ]
    effect = compute_raw_effect(records)
    assert effect["n_records"] == 3
    assert effect["n_parsed"] == 2
    assert effect["mean_lift_steps"] == 300.0
    assert effect["n_lift_cells"] == 1


def test_compute_raw_effect_skips_partial_histories() -> None:
    partial = {
        "state": _state("low", "none"),
        "action": CONTROL_ACTION,
        "content": "\n".join(
            f'{{"day": {d}, "morning_steps": 1500, "afternoon_steps": 1500}}'
            for d in range(1, 4)
        ),
    }
    records = [partial]
    effect = compute_raw_effect(records)
    assert effect["n_records"] == 1
    assert effect["n_parsed"] == 0
    assert effect["mean_lift_steps"] is None
