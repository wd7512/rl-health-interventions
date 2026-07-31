"""Tests for llm_bootstrapping.parse_pearl module."""

from __future__ import annotations

import json

from rl_health_interventions.llm_bootstrapping.parse_pearl import (
    bin_morning_ratio,
    bin_recent_steps_mean,
    bin_walk_pattern,
    history_to_factors,
    parse_day_history,
)


class TestParseDayHistory:
    def test_valid_7_days(self) -> None:
        lines = [
            json.dumps(
                {
                    "day": i,
                    "morning_steps": 2000 + i * 100,
                    "afternoon_steps": 1500 + i * 50,
                }
            )
            for i in range(1, 8)
        ]
        result = parse_day_history("\n".join(lines))
        assert result is not None
        assert len(result) == 7
        assert result[0]["morning_steps"] == 2100  # 2000 + 1*100
        assert result[6]["afternoon_steps"] == 1850  # 1500 + 7*50

    def test_single_day_rejected(self) -> None:
        result = parse_day_history(
            '{"day": 1, "morning_steps": 3000, "afternoon_steps": 2000}'
        )
        assert result is None

    def test_missing_morning_steps(self) -> None:
        result = parse_day_history('{"day": 1, "afternoon_steps": 2000}')
        assert result is None

    def test_missing_afternoon_steps(self) -> None:
        result = parse_day_history('{"day": 1, "morning_steps": 3000}')
        assert result is None

    def test_missing_day(self) -> None:
        result = parse_day_history('{"morning_steps": 3000, "afternoon_steps": 2000}')
        assert result is None

    def test_negative_steps(self) -> None:
        result = parse_day_history(
            '{"day": 1, "morning_steps": -1, "afternoon_steps": 2000}'
        )
        assert result is None

    def test_float_steps(self) -> None:
        lines = [
            json.dumps(
                {
                    "day": i,
                    "morning_steps": 2500.5,
                    "afternoon_steps": 1500.2,
                }
            )
            for i in range(1, 8)
        ]
        result = parse_day_history("\n".join(lines))
        assert result is not None
        assert result[0]["morning_steps"] == 2500

    def test_empty_string(self) -> None:
        result = parse_day_history("")
        assert result is None

    def test_no_json(self) -> None:
        result = parse_day_history("not json at all")
        assert result is None

    def test_partial_history_rejected(self) -> None:
        lines = [
            json.dumps(
                {
                    "day": i,
                    "morning_steps": 2000,
                    "afternoon_steps": 1500,
                }
            )
            for i in range(1, 6)
        ]
        result = parse_day_history("\n".join(lines))
        assert result is None

    def test_duplicate_days_rejected(self) -> None:
        lines = [
            json.dumps(
                {
                    "day": 1 if i % 2 else 2,
                    "morning_steps": 2000,
                    "afternoon_steps": 1500,
                }
            )
            for i in range(7)
        ]
        result = parse_day_history("\n".join(lines))
        assert result is None

    def test_extra_text_around_json(self) -> None:
        lines = [
            json.dumps(
                {
                    "day": i,
                    "morning_steps": 2000,
                    "afternoon_steps": 1500,
                }
            )
            for i in range(1, 8)
        ]
        text = "Here is the history:\n" + "\n".join(lines) + "\nDone."
        result = parse_day_history(text)
        assert result is not None
        assert len(result) == 7

    def test_fenced_json_block(self) -> None:
        lines = [
            json.dumps(
                {
                    "day": i,
                    "morning_steps": 2000,
                    "afternoon_steps": 1500,
                }
            )
            for i in range(1, 8)
        ]
        text = "```json\n" + "\n".join(lines) + "\n```"
        result = parse_day_history(text)
        assert result is not None
        assert len(result) == 7

    def test_custom_expected_days(self) -> None:
        result = parse_day_history(
            '{"day": 1, "morning_steps": 3000, "afternoon_steps": 2000}',
            expected_days=1,
        )
        assert result is not None
        assert len(result) == 1


class TestBinRecentStepsMean:
    def test_low(self) -> None:
        assert bin_recent_steps_mean([3000, 3500, 2800]) == "low"

    def test_moderate(self) -> None:
        assert bin_recent_steps_mean([5000, 5500, 6000]) == "moderate"

    def test_high(self) -> None:
        assert bin_recent_steps_mean([8000, 7500, 9000]) == "high"

    def test_boundary_low_moderate(self) -> None:
        assert bin_recent_steps_mean([4000]) == "moderate"

    def test_boundary_moderate_high(self) -> None:
        assert bin_recent_steps_mean([7000]) == "moderate"

    def test_empty(self) -> None:
        assert bin_recent_steps_mean([]) == "moderate"


class TestBinWalkPattern:
    def test_low(self) -> None:
        assert bin_walk_pattern([3000, 3500, 2800]) == "low"

    def test_high(self) -> None:
        assert bin_walk_pattern([5500, 6000, 5000]) == "high"

    def test_boundary(self) -> None:
        assert bin_walk_pattern([5000]) == "high"

    def test_empty(self) -> None:
        assert bin_walk_pattern([]) == "low"


class TestBinMorningRatio:
    def test_evening_bias(self) -> None:
        # 2000 morning / 6000 total = 0.33 → < 0.4 → evening
        assert bin_morning_ratio([2000], [6000]) == "evening"

    def test_balanced(self) -> None:
        # 3000 morning / 6000 total = 0.5 → balanced
        assert bin_morning_ratio([3000], [6000]) == "balanced"

    def test_morning_bias(self) -> None:
        # 5000 morning / 8000 total = 0.625 → > 0.6 → morning
        assert bin_morning_ratio([5000], [8000]) == "morning"

    def test_zero_total(self) -> None:
        assert bin_morning_ratio([0], [0]) == "balanced"

    def test_empty(self) -> None:
        assert bin_morning_ratio([], []) == "balanced"


class TestHistoryToFactors:
    def test_low_steps_low_walk(self) -> None:
        history = [
            {"day": i, "morning_steps": 1500, "afternoon_steps": 1000}
            for i in range(1, 8)
        ]
        factors = history_to_factors(history)
        assert factors["recent_steps_mean"] == "low"
        assert factors["recent_walk_pattern"] == "low"

    def test_moderate_steps_high_walk(self) -> None:
        history = [
            {"day": i, "morning_steps": 3000, "afternoon_steps": 2500}
            for i in range(1, 8)
        ]
        factors = history_to_factors(history)
        assert factors["recent_steps_mean"] == "moderate"
        assert factors["recent_walk_pattern"] == "high"

    def test_high_steps(self) -> None:
        history = [
            {"day": i, "morning_steps": 4500, "afternoon_steps": 4000}
            for i in range(1, 8)
        ]
        factors = history_to_factors(history)
        assert factors["recent_steps_mean"] == "high"
        assert factors["recent_walk_pattern"] == "high"

    def test_morning_ratio_evening(self) -> None:
        # 1000 morning, 4000 total -> 0.25 -> < 0.4 -> evening
        history = [
            {"day": i, "morning_steps": 1000, "afternoon_steps": 3000}
            for i in range(1, 8)
        ]
        factors = history_to_factors(history)
        assert factors["morning_steps_ratio"] == "evening"

    def test_morning_ratio_balanced(self) -> None:
        # 2500 morning, 5000 total -> 0.5 -> balanced
        history = [
            {"day": i, "morning_steps": 2500, "afternoon_steps": 2500}
            for i in range(1, 8)
        ]
        factors = history_to_factors(history)
        assert factors["morning_steps_ratio"] == "balanced"

    def test_morning_ratio_morning(self) -> None:
        # 4000 morning, 5000 total -> 0.8 -> > 0.6 -> morning
        history = [
            {"day": i, "morning_steps": 4000, "afternoon_steps": 1000}
            for i in range(1, 8)
        ]
        factors = history_to_factors(history)
        assert factors["morning_steps_ratio"] == "morning"

    def test_returns_all_three_factors(self) -> None:
        history = [
            {"day": i, "morning_steps": 2500, "afternoon_steps": 2500}
            for i in range(1, 8)
        ]
        factors = history_to_factors(history)
        assert set(factors.keys()) == {
            "recent_steps_mean",
            "recent_walk_pattern",
            "morning_steps_ratio",
        }
