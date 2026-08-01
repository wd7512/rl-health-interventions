"""Unit tests for generate_pearl_full chunking / resume / finalize."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pearl_recalibration.generate_pearl_full import (  # noqa: E402
    _completed_states,
    _finalize,
    _load_raw_records,
    _state_key,
)


def _state(rsm: str) -> dict:
    return {
        "recent_steps_mean": rsm,
        "recent_walk_pattern": "low",
        "morning_steps_ratio": "balanced",
        "day_of_week": "weekday",
        "burden": "none",
    }


def _day_response() -> str:
    return "\n".join(
        f'{{"day": {d}, "morning_steps": 4000, "afternoon_steps": 4000}}'
        for d in range(1, 8)
    )


def _write_raw(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_state_key_round_trip() -> None:
    assert _state_key(_state("high")) == _state_key(_state("high"))
    assert _state_key(_state("high")) != _state_key(_state("low"))


def test_load_raw_records_skips_blank_and_malformed(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text('{"a": 1}\n\nnot-json\n{"b": 2}\n')
    records = _load_raw_records(path)
    assert len(records) == 2


def test_completed_states_requires_all_actions_and_samples(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    state = _state("low")
    _write_raw(
        path,
        [
            {
                "state": state,
                "action": f"theme_{i}",
                "content": _day_response(),
            }
            for i in range(13)
            for _ in range(10)
        ],
    )
    assert _state_key(_state("low")) in _completed_states(path)
    assert _state_key(_state("high")) not in _completed_states(path)
    assert len(_completed_states(path)) == 1


def test_completed_states_requires_full_sample_count(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    state = _state("low")
    _write_raw(path, [{"state": state, "action": "idle", "content": _day_response()}])
    assert _state_key(state) not in _completed_states(path)


def test_finalize_aggregates_raw_into_table(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    state = _state("low")
    records = [
        {"state": state, "action": "idle", "content": _day_response()} for _ in range(3)
    ]
    _write_raw(raw, records)
    out_path = tmp_path / "pearl_bootstrap.json"
    table = _finalize(raw, out_path=out_path)
    assert table == out_path
    loaded = json.loads(table.read_text())
    assert len(loaded["transitions"]) == 1
    probs = loaded["transitions"][0]["next_state_probs"]["recent_steps_mean"]
    assert abs(sum(probs.values()) - 1.0) <= 1e-6


def test_finalize_raises_on_empty_raw(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    _write_raw(raw, [])
    with pytest.raises(ValueError, match="nothing to aggregate"):
        _finalize(raw)
