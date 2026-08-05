"""Unit tests for the shared PEARL table aggregator."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl_health_interventions.llm_bootstrapping.table_aggregate import (  # noqa: E402
    MIN_SAMPLES_PER_CELL,
    aggregate_to_table,
)


def _state(rsm: str) -> dict:
    return {
        "recent_steps_mean": rsm,
        "recent_walk_pattern": "low",
        "morning_steps_ratio": "balanced",
        "day_of_week": "weekday",
        "burden": "none",
    }


def _day_response(*totals: tuple[int, int]) -> str:
    return "\n".join(
        f'{{"day": {d}, "morning_steps": {m}, "afternoon_steps": {a}}}'
        for d, (m, a) in enumerate(totals, start=1)
    )


def _low_response() -> str:
    return _day_response(*[(1500, 1500)] * 7)


def _moderate_response() -> str:
    return _day_response(*[(3000, 3000)] * 7)


def _high_response() -> str:
    return _day_response(*[(4000, 4000)] * 7)


def _round_trip(table: dict) -> None:
    """Sum-to-1 invariant: every factor distribution sums to 1.0 (4dp)."""
    for t in table["transitions"]:
        for probs in t["next_state_probs"].values():
            assert abs(sum(probs.values()) - 1.0) <= 1e-6
            assert min(probs.values()) >= 0.0


def test_aggregates_recent_steps_mean_three_levels() -> None:
    results = [
        {"content": _low_response()},
        {"content": _moderate_response()},
        {"content": _high_response()},
    ]
    pairs = [(_state("low"), "idle")] * 3
    table = aggregate_to_table(results, pairs)
    probs = table["transitions"][0]["next_state_probs"]["recent_steps_mean"]
    assert abs(sum(probs.values()) - 1.0) <= 1e-6
    assert probs == {"high": 0.3333, "low": 0.3333, "moderate": 0.3334}
    assert table["transitions"][0]["n_samples"] == 3
    _round_trip(table)


def test_sum_to_one_for_non_divisible_counts() -> None:
    results = [{"content": _low_response()}] * 2 + [{"content": _high_response()}]
    pairs = [(_state("low"), "idle")] * 3
    table = aggregate_to_table(results, pairs)
    probs = table["transitions"][0]["next_state_probs"]["recent_steps_mean"]
    assert probs["low"] == 0.6667
    assert probs["high"] == 0.3333
    assert abs(sum(probs.values()) - 1.0) <= 1e-6
    _round_trip(table)


def test_sum_to_one_at_even_three_way_split() -> None:
    results = [
        {"content": _low_response()},
        {"content": _moderate_response()},
        {"content": _high_response()},
        {"content": _low_response()},
        {"content": _moderate_response()},
        {"content": _high_response()},
    ]
    pairs = [(_state("low"), "idle")] * 6
    table = aggregate_to_table(results, pairs)
    probs = table["transitions"][0]["next_state_probs"]["recent_steps_mean"]
    assert abs(sum(probs.values()) - 1.0) <= 1e-6
    _round_trip(table)


def test_errors_and_unparseable_are_skipped() -> None:
    results = [
        {"error": "boom"},
        {"content": "not a valid history"},
        {"content": _high_response()},
    ]
    pairs = [(_state("low"), "idle")] * 3
    table = aggregate_to_table(results, pairs)
    assert table["transitions"] == []
    table = aggregate_to_table(results, pairs, min_samples_per_cell=1)
    assert len(table["transitions"]) == 1
    assert table["transitions"][0]["n_samples"] == 1


def test_min_samples_per_cell_drop() -> None:
    results = [{"content": _high_response()}]
    pairs = [(_state("low"), "idle")]
    table = aggregate_to_table(results, pairs)
    assert table["transitions"] == []
    table = aggregate_to_table(results, pairs, min_samples_per_cell=1)
    assert len(table["transitions"]) == 1


def test_unknown_factor_values_are_tolerated() -> None:
    high = _high_response()
    results = [
        {"content": high},
        {"content": high},
        {"content": high},
    ]
    pairs = [(_state("low"), "idle")] * 3
    table = aggregate_to_table(results, pairs)
    probs = table["transitions"][0]["next_state_probs"]
    assert set(probs) == {
        "recent_steps_mean",
        "recent_walk_pattern",
        "morning_steps_ratio",
    }
    _round_trip(table)


def test_min_samples_constant_is_defined() -> None:
    assert MIN_SAMPLES_PER_CELL == 2
