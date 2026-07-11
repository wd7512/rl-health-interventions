"""Tests for llm_bootstrapping.parse module."""

from __future__ import annotations

from rl_health_interventions.llm_bootstrapping.parse import (
    bin_step_count,
    extract_json,
    parse_day_boundary,
    parse_within_day,
)


class TestBinStepCount:
    def test_inactive_below_800(self) -> None:
        assert bin_step_count(0) == "inactive"
        assert bin_step_count(399) == "inactive"
        assert bin_step_count(799) == "inactive"

    def test_boundary_800_is_moderate(self) -> None:
        assert bin_step_count(800) == "moderate"

    def test_moderate_range(self) -> None:
        assert bin_step_count(801) == "moderate"
        assert bin_step_count(1200) == "moderate"
        assert bin_step_count(1600) == "moderate"

    def test_active_above_1600(self) -> None:
        assert bin_step_count(1601) == "active"
        assert bin_step_count(10000) == "active"


class TestExtractJson:
    def test_simple_json(self) -> None:
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_no_braces(self) -> None:
        assert extract_json("hello world") is None

    def test_only_opening_brace(self) -> None:
        assert extract_json('{"key":') is None

    def test_only_closing_brace(self) -> None:
        assert extract_json("hello}") is None

    def test_malformed_json(self) -> None:
        assert extract_json('{"key": invalid}') is None

    def test_last_json_object(self) -> None:
        text = '{"first": 1} some text {"second": 2}'
        result = extract_json(text)
        assert result == {"second": 2}

    def test_json_with_thinking_text(self) -> None:
        text = 'Let me think... {"working": true}\n{"sleep_quality": "good"}'
        result = extract_json(text)
        assert result == {"sleep_quality": "good"}

    def test_extra_trailing_brace(self) -> None:
        text = '{"key": "value"}}'
        result = extract_json(text)
        assert result is None

    def test_empty_object(self) -> None:
        result = extract_json("{}")
        assert result == {}


class TestParseDayBoundary:
    def test_good_sleep(self) -> None:
        assert parse_day_boundary('{"sleep_quality": "good"}') == "good"

    def test_poor_sleep(self) -> None:
        assert parse_day_boundary('{"sleep_quality": "poor"}') == "poor"

    def test_invalid_sleep_quality(self) -> None:
        assert parse_day_boundary('{"sleep_quality": "unknown"}') is None

    def test_missing_key(self) -> None:
        assert parse_day_boundary('{"foo": "bar"}') is None

    def test_non_json(self) -> None:
        assert parse_day_boundary("not json") is None

    def test_empty_string(self) -> None:
        assert parse_day_boundary("") is None

    def test_whitespace_only(self) -> None:
        assert parse_day_boundary("  ") is None


class TestParseWithinDay:
    def test_valid_inactive(self) -> None:
        result = parse_within_day('{"steps": 400, "step_bin": "inactive"}')
        assert result == (400, "inactive")

    def test_valid_moderate(self) -> None:
        result = parse_within_day('{"steps": 1200, "step_bin": "moderate"}')
        assert result == (1200, "moderate")

    def test_valid_active(self) -> None:
        result = parse_within_day('{"steps": 2000, "step_bin": "active"}')
        assert result == (2000, "active")

    def test_step_bin_mismatch(self) -> None:
        result = parse_within_day('{"steps": 2000, "step_bin": "inactive"}')
        assert result is not None
        steps, computed = result
        assert steps == 2000
        assert computed == "active"

    def test_missing_steps(self) -> None:
        assert parse_within_day('{"step_bin": "inactive"}') is None

    def test_missing_step_bin(self) -> None:
        assert parse_within_day('{"steps": 400}') is None

    def test_negative_steps(self) -> None:
        assert parse_within_day('{"steps": -1, "step_bin": "inactive"}') is None

    def test_invalid_step_bin(self) -> None:
        assert parse_within_day('{"steps": 400, "step_bin": "unknown"}') is None

    def test_non_dict(self) -> None:
        assert parse_within_day("hello") is None

    def test_zero_steps(self) -> None:
        result = parse_within_day('{"steps": 0, "step_bin": "inactive"}')
        assert result == (0, "inactive")

    def test_step_bin_mismatch_moderate(self) -> None:
        result = parse_within_day('{"steps": 400, "step_bin": "moderate"}')
        assert result is not None
        steps, computed = result
        assert steps == 400
        assert computed == "inactive"
