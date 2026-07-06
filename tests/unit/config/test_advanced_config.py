from __future__ import annotations

import pytest
from pydantic import ValidationError

from rl_health_interventions.config.schemas import MDPConfig


def _base_raw() -> dict:
    return {
        "episode_days": 1,
        "steps_per_day": 1,
        "seed": 42,
        "state": {"variables": {"x": {"names": ["a", "b"]}}},
        "initial_state": {"x": "a"},
        "actions": ["go"],
        "reward": {
            "variables": {"v": {"source": "state.x", "mapping": {"a": 0.0, "b": 1.0}}},
            "formula": "v",
        },
        "transition_model": {"type": "random"},
        "agents": [],
    }


class TestCyclicAdvance:
    def test_valid_pattern(self) -> None:
        raw = _base_raw()
        raw["state"]["variables"]["x"]["advanced"] = {
            "type": "cyclic",
            "granularity": "daily",
            "pattern": ["a", "b"],
        }
        MDPConfig.model_validate(raw)

    def test_invalid_pattern_value(self) -> None:
        raw = _base_raw()
        raw["state"]["variables"]["x"]["advanced"] = {
            "type": "cyclic",
            "granularity": "daily",
            "pattern": ["a", "c"],
        }
        with pytest.raises(ValidationError, match="not in variable names"):
            MDPConfig.model_validate(raw)


class TestRollingWindowAdvance:
    def _raw_with_rolling(self, mapping: dict, window_size: int = 2) -> dict:
        raw = _base_raw()
        raw["state"]["variables"]["x"]["names"] = ["a", "b", "c"]
        raw["reward"]["variables"]["v"]["mapping"] = {"a": 0.0, "b": 0.5, "c": 1.0}
        raw["state"]["variables"]["x"]["advanced"] = {
            "type": "rolling_window_count",
            "window_size": window_size,
            "conditions": [{"factor": "action", "type": "in", "values": ["go"]}],
            "mapping": mapping,
        }
        return raw

    def test_valid_mapping(self) -> None:
        raw = self._raw_with_rolling({0: "a", 1: "b", 2: "c"})
        MDPConfig.model_validate(raw)

    def test_missing_mapping_keys(self) -> None:
        raw = self._raw_with_rolling({0: "a", 2: "c"})
        with pytest.raises(ValidationError, match="must cover"):
            MDPConfig.model_validate(raw)

    def test_extra_mapping_keys(self) -> None:
        raw = self._raw_with_rolling({0: "a", 1: "b", 2: "c", 3: "a"})
        with pytest.raises(ValidationError, match="must cover"):
            MDPConfig.model_validate(raw)

    def test_mapping_value_not_in_names(self) -> None:
        raw = self._raw_with_rolling({0: "a", 1: "b", 2: "z"})
        with pytest.raises(ValidationError, match="not in variable names"):
            MDPConfig.model_validate(raw)

    def test_invalid_condition_factor(self) -> None:
        raw = _base_raw()
        raw["state"]["variables"]["x"]["names"] = ["a", "b", "c"]
        raw["reward"]["variables"]["v"]["mapping"] = {"a": 0.0, "b": 0.5, "c": 1.0}
        raw["state"]["variables"]["x"]["advanced"] = {
            "type": "rolling_window_count",
            "window_size": 2,
            "conditions": [{"factor": "nonexistent", "type": "in", "values": ["go"]}],
            "mapping": {0: "a", 1: "b", 2: "c"},
        }
        with pytest.raises(ValidationError, match="not a declared variable"):
            MDPConfig.model_validate(raw)

    def test_condition_type_must_be_in(self) -> None:
        raw = _base_raw()
        raw["state"]["variables"]["x"]["names"] = ["a", "b", "c"]
        raw["reward"]["variables"]["v"]["mapping"] = {"a": 0.0, "b": 0.5, "c": 1.0}
        raw["state"]["variables"]["x"]["advanced"] = {
            "type": "rolling_window_count",
            "window_size": 2,
            "conditions": [{"factor": "action", "type": "in", "values": ["go"]}],
            "mapping": {0: "a", 1: "b", 2: "c"},
        }
        # Valid config should pass (type "in" is accepted)
        MDPConfig.model_validate(raw)

    def test_window_size_must_be_positive(self) -> None:
        raw = self._raw_with_rolling({0: "a", 1: "b", 2: "c"}, window_size=0)
        with pytest.raises(ValidationError, match="greater than"):
            MDPConfig.model_validate(raw)


class TestCyclicAdvanceConstraints:
    def test_empty_pattern_rejected(self) -> None:
        raw = _base_raw()
        raw["state"]["variables"]["x"]["advanced"] = {
            "type": "cyclic",
            "granularity": "daily",
            "pattern": [],
        }
        with pytest.raises(ValidationError, match="too_short"):
            MDPConfig.model_validate(raw)
