"""Unit tests for generate_pearl_full cell-level resume / finalize."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl_health_interventions.llm_bootstrapping.prompts.pearl import (  # noqa: E402
    ACTIONS,
)
from scripts.pearl_recalibration.generate_pearl_full import (  # noqa: E402
    _content_counts,
    _finalize,
    _load_raw_records,
    _state_key,
    _strip_errors,
    _todo_cells,
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


def _full_state_records(state: dict, samples: int) -> list[dict]:
    return [
        {"state": state, "action": action, "content": _day_response()}
        for action in ACTIONS
        for _ in range(samples)
    ]


def test_todo_cells_empty_when_state_complete(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    _write_raw(path, _full_state_records(_state("low"), samples=10))
    todo = _todo_cells(path, samples=10)
    assert all(_state_key(state) != _state_key(_state("low")) for state, _a, _n in todo)
    assert any(
        _state_key(state) == _state_key(_state("high")) for state, _a, _n in todo
    )


def test_todo_cells_reports_shortfall(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    state = _state("low")
    records = _full_state_records(state, samples=7)
    records.append({"state": state, "action": "idle", "content": _day_response()})
    _write_raw(path, records)
    todo = _todo_cells(path, samples=10)
    idle = next(
        n for s, a, n in todo if _state_key(s) == _state_key(state) and a == "idle"
    )
    physical = next(
        n
        for s, a, n in todo
        if _state_key(s) == _state_key(state) and a == "physical_opportunity_morning"
    )
    assert idle == 2
    assert physical == 3


def test_todo_cells_errors_do_not_count(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    state = _state("low")
    records = _full_state_records(state, samples=10)
    records.append({"state": state, "action": "idle", "error": "rate-limited"})
    _write_raw(path, records)
    todo = _todo_cells(path, samples=10)
    assert not any(
        _state_key(s) == _state_key(state) and a == "idle" for s, a, _n in todo
    )


def test_todo_cells_empty_file_needs_full_samples(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    _write_raw(path, [])
    todo = _todo_cells(path, samples=10)
    assert len(todo) == 108 * len(ACTIONS)
    assert all(n == 10 for _s, _a, n in todo)


def test_content_counts_ignores_errors(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    state = _state("low")
    _write_raw(
        path,
        [
            {"state": state, "action": "idle", "content": _day_response()},
            {"state": state, "action": "idle", "error": "rate-limited"},
        ],
    )
    counts = _content_counts(path)
    assert counts[(_state_key(state), "idle")] == 1


def test_strip_errors_removes_error_records(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    state = _state("low")
    _write_raw(
        path,
        [
            {"state": state, "action": "idle", "content": _day_response()},
            {"state": state, "action": "idle", "error": "rate-limited"},
        ],
    )
    _strip_errors(path)
    records = _load_raw_records(path)
    assert len(records) == 1
    assert "content" in records[0]


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


def test_flush_batch_appends_primary_before_retry(tmp_path: Path, monkeypatch) -> None:
    """Primary records hit disk even when the retry call crashes."""
    from scripts.pearl_recalibration import generate_pearl_full as gp

    raw = tmp_path / "raw.jsonl"
    state = _state("low")
    batch = [
        (_day_response(), state, "idle"),
        ("garbage-not-json", state, "physical_opportunity_morning"),
        (_day_response(), state, "emotional_regulation"),
    ]
    calls = {"n": 0}

    def fake_batch_complete(prompts, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"content": p} for p in prompts]
        msg = "simulated retry stall"
        raise TimeoutError(msg)

    monkeypatch.setattr(gp, "batch_complete", fake_batch_complete)
    with pytest.raises(TimeoutError, match="simulated retry stall"):
        gp._flush_batch(
            raw,
            "sys",
            batch,
            temperature=0.3,
            max_workers=2,
            total_prompts=3,
            done_prompts=0,
            start=gp.datetime.now(gp.UTC),
        )
    records = _load_raw_records(raw)
    assert len(records) == 3
    assert all("content" in r for r in records)
