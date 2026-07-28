from __future__ import annotations

import json
from pathlib import Path

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_BASIC_DIR = _FIXTURES_DIR / "basic"
_NO_GLOBAL_STATE_DIR = _FIXTURES_DIR / "no_global_state"
_SPRINT1_DIR = _FIXTURES_DIR / "sprint1"
_PEARL_DIR = _FIXTURES_DIR / "pearl"
_INVALID_DIR = _FIXTURES_DIR / "invalid"
_EDGE_DIR = _FIXTURES_DIR / "edge_cases"
_BASIC_MULTI_DIR = _FIXTURES_DIR / "basic_multi"
_THREE_FACTORS_DIR = _FIXTURES_DIR / "three_factors"
_TWO_FACTORS_DIR = _FIXTURES_DIR / "two_factors"


class TestFixturesValidity:
    """Verify that the test fixture JSON files are themselves valid
    according to the new format (``global_state`` + ``transitions``)."""

    # ── basic/ ──────────────────────────────────────────────────────

    def test_basic_table_json_transitions(self) -> None:
        data = json.loads((_BASIC_DIR / "table.json").read_text(encoding="utf-8"))
        assert "transitions" in data
        assert isinstance(data["transitions"], list)
        assert len(data["transitions"]) == 4
        if "global_state" in data:
            assert isinstance(data["global_state"], dict)

    def test_basic_no_global_state_missing_key_is_ok(self) -> None:
        data = json.loads(
            (_NO_GLOBAL_STATE_DIR / "no_global_state.json").read_text(encoding="utf-8")
        )
        assert "transitions" in data
        assert "global_state" not in data

    def test_basic_empty_transitions_array(self) -> None:
        data = json.loads(
            (_BASIC_DIR / "empty_transitions.json").read_text(encoding="utf-8")
        )
        assert data["transitions"] == []

    def test_basic_two_factors_each_distribution_sums_to_one(self) -> None:
        data = json.loads(
            (_TWO_FACTORS_DIR / "two_factors.json").read_text(encoding="utf-8")
        )
        for entry in data["transitions"]:
            nsp = entry.get("next_state_probs", {})
            for factor, dist in nsp.items():
                total = sum(dist.values())
                assert abs(total - 1.0) < 1e-6, (
                    f"Factor {factor!r} sum={total} in entry state={entry['state']} "
                    f"action={entry['action']}"
                )

    def test_basic_three_factors_each_distribution_sums_to_one(self) -> None:
        data = json.loads(
            (_THREE_FACTORS_DIR / "three_factors.json").read_text(encoding="utf-8")
        )
        for entry in data["transitions"]:
            nsp = entry.get("next_state_probs", {})
            assert len(nsp) == 3  # x, y, z
            for factor, dist in nsp.items():
                total = sum(dist.values())
                assert abs(total - 1.0) < 1e-6, (
                    f"Factor {factor!r} sum={total} in entry state={entry['state']} "
                    f"action={entry['action']}"
                )

    # ── basic_multi/ ────────────────────────────────────────────────

    def test_basic_multi_both_files_have_expected_entries(self) -> None:
        for i in range(2):
            path = _BASIC_MULTI_DIR / f"step_{i}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert len(data["transitions"]) == 2
            assert data["global_state"]["step_of_day"] == i

    # ── sprint1/ ────────────────────────────────────────────────────

    def test_sprint1_fixtures_have_step_of_day(self) -> None:
        for i in range(2):
            path = _SPRINT1_DIR / f"step_{i}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["global_state"]["step_of_day"] == i

    def test_sprint1_all_entries_have_both_factors(self) -> None:
        for i in range(2):
            path = _SPRINT1_DIR / f"step_{i}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data["transitions"]:
                nsp = entry.get("next_state_probs", {})
                assert "step_bin" in nsp, (
                    f"step_{i}.json entry missing step_bin: "
                    f"state={entry['state']} action={entry['action']}"
                )
                assert "sleep" in nsp, (
                    f"step_{i}.json entry missing sleep: "
                    f"state={entry['state']} action={entry['action']}"
                )
                for dist in nsp.values():
                    assert abs(sum(dist.values()) - 1.0) < 1e-6

    # ── pearl/ ──────────────────────────────────────────────────────

    def test_pearl_fixture_all_entries_have_three_factors(self) -> None:
        data = json.loads((_PEARL_DIR / "transition.json").read_text(encoding="utf-8"))
        for entry in data["transitions"]:
            nsp = entry.get("next_state_probs", {})
            assert "engagement" in nsp
            assert "mood" in nsp
            assert "social" in nsp
            for dist in nsp.values():
                assert abs(sum(dist.values()) - 1.0) < 1e-6

    # ── invalid/ ────────────────────────────────────────────────────

    def test_invalid_missing_transitions_key(self) -> None:
        data = json.loads(
            (_INVALID_DIR / "missing_transitions.json").read_text(encoding="utf-8")
        )
        assert "transitions" not in data
        assert "global_state" in data

    def test_invalid_bad_probability_sum(self) -> None:
        data = json.loads(
            (_INVALID_DIR / "bad_probability_sum.json").read_text(encoding="utf-8")
        )
        entry = data["transitions"][0]
        total = sum(entry["next_state_probs"]["activity"].values())
        assert abs(total - 1.0) >= 1e-6  # Should NOT sum to 1

    def test_invalid_negative_probability(self) -> None:
        data = json.loads(
            (_INVALID_DIR / "negative_probability.json").read_text(encoding="utf-8")
        )
        dist = data["transitions"][0]["next_state_probs"]["activity"]
        assert any(v < 0 for v in dist.values())

    def test_invalid_probability_over_one(self) -> None:
        data = json.loads(
            (_INVALID_DIR / "probability_over_one.json").read_text(encoding="utf-8")
        )
        dist = data["transitions"][0]["next_state_probs"]["activity"]
        assert any(v > 1.0 for v in dist.values())

    def test_invalid_missing_next_state_probs(self) -> None:
        data = json.loads(
            (_INVALID_DIR / "missing_next_state_probs.json").read_text(encoding="utf-8")
        )
        assert "next_state_probs" not in data["transitions"][0]

    def test_invalid_missing_state_action_probs(self) -> None:
        data = json.loads(
            (_INVALID_DIR / "missing_state_action_probs.json").read_text(
                encoding="utf-8"
            )
        )
        assert "state" not in data["transitions"][0]

    # ── edge_cases/ ─────────────────────────────────────────────────

    def test_edge_empty_transitions(self) -> None:
        data = json.loads((_EDGE_DIR / "empty.json").read_text(encoding="utf-8"))
        assert data["transitions"] == []

    def test_edge_fixtures_exist(self) -> None:
        for name in (
            "empty",
            "missing_action",
            "reproducibility",
            "unknown_factor_value",
            "uploaded_state_not_in_table",
        ):
            path = _EDGE_DIR / f"{name}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert isinstance(data, dict)
