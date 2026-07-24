from __future__ import annotations

import itertools
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions.table_transition import TableTransition

_TABLES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "tables"
    / "persona"
    / "base_deepseek-v4-flash"
)


def _sprint1_config(table_dir: str | None = None, seed: int = 42) -> MDPConfig:
    return MDPConfig(
        episode_days=1,
        steps_per_day=5,
        seed=seed,
        state={
            "variables": {
                "step_bin": {"names": ["inactive", "moderate", "active"]},
                "sleep": {"names": ["good", "poor"]},
                "day_of_week": {
                    "names": ["weekday", "weekend"],
                    "advanced": {
                        "type": "cyclic",
                        "granularity": "daily",
                        "pattern": [
                            "weekday",
                            "weekday",
                            "weekday",
                            "weekday",
                            "weekday",
                            "weekend",
                            "weekend",
                        ],
                    },
                },
                "burden": {
                    "names": ["low", "medium", "high"],
                    "advanced": {
                        "type": "rolling_window_count",
                        "window_size": 3,
                        "conditions": [
                            {
                                "factor": "action",
                                "type": "in",
                                "values": [
                                    "movement_suggestion",
                                    "goal_reminder",
                                    "journal",
                                ],
                            }
                        ],
                        "mapping": {0: "low", 1: "medium", 2: "high", 3: "high"},
                    },
                },
            }
        },
        initial_state={
            "step_bin": "inactive",
            "sleep": "good",
            "day_of_week": "weekday",
            "burden": "low",
        },
        actions=["idle", "movement_suggestion", "goal_reminder", "journal"],
        reward={
            "constants": {"alpha": 0.9},
            "variables": {
                "step_bin_value": {
                    "source": "state.step_bin",
                    "mapping": {"inactive": 0.0, "moderate": 0.5, "active": 1.0},
                },
                "sleep_value": {
                    "source": "state.sleep",
                    "mapping": {"good": 1.0, "poor": -1.0},
                },
                "action_penalty": {
                    "source": "action",
                    "mapping": {
                        "idle": 0.0,
                        "movement_suggestion": 0.05,
                        "goal_reminder": 0.05,
                        "journal": 0.05,
                    },
                },
            },
            "formula": (
                "alpha * step_bin_value + (1 - alpha) * sleep_value - action_penalty"
            ),
        },
        transition_model={
            "type": "table_transition",
            "table_dir": table_dir or str(_TABLES_DIR),
        },
        agents=[],
    )


class TestTableLoading:
    def test_loads_all_tables(self) -> None:
        t = TableTransition(_sprint1_config(), seed=42)
        assert len(t.day_boundary) > 0
        assert len(t.within_day) == 5
        for i, wd in enumerate(t.within_day):
            assert len(wd) > 0, f"within_day_{i} is empty"
        assert "active|high|weekend|poor" in t.day_boundary

    def test_table_structure(self) -> None:
        t = TableTransition(_sprint1_config(), seed=42)
        for key, (targets, probs) in t.day_boundary.items():
            assert len(targets) == len(probs)
            assert abs(float(probs.sum()) - 1.0) < 1e-6
            assert np.all(probs > 0), f"{key}: has non-positive probability"
        for i, wd in enumerate(t.within_day):
            for key, (targets, probs) in wd.items():
                assert len(targets) == len(probs)
                assert abs(float(probs.sum()) - 1.0) < 1e-6
                assert np.all(probs > 0), f"within_day_{i}: {key} non-positive prob"


class TestKeyFormat:
    def test_day_boundary_key_format(self) -> None:
        t = TableTransition(_sprint1_config(), seed=42)
        state = StateView(
            factors={
                "step_bin": "inactive",
                "sleep": "good",
                "day_of_week": "weekday",
                "burden": "low",
            },
            day=0,
            step_of_day=0,
        )
        key = t._build_state_key(state, "idle", for_within_day=False)
        assert key == "inactive|low|weekday|good"

    def test_within_day_key_format(self) -> None:
        t = TableTransition(_sprint1_config(), seed=42)
        state = StateView(
            factors={
                "step_bin": "moderate",
                "sleep": "poor",
                "day_of_week": "weekend",
                "burden": "high",
            },
            day=0,
            step_of_day=2,
        )
        key = t._build_state_key(state, "movement_suggestion", for_within_day=True)
        assert key == "moderate|high|movement_suggestion|weekend|poor"


class TestRouting:
    def test_step_zero_samples_sleep_and_step_bin(self) -> None:
        t = TableTransition(_sprint1_config(), seed=42)
        state = StateView(
            factors={
                "step_bin": "inactive",
                "sleep": "good",
                "day_of_week": "weekday",
                "burden": "low",
            },
            day=0,
            step_of_day=0,
        )
        updates = t.transition(state, "idle")
        assert "sleep" in updates
        assert "step_bin" in updates
        assert updates["sleep"] in ("good", "poor")
        assert updates["step_bin"] in ("inactive", "moderate", "active")

    def test_step_nonzero_samples_only_step_bin(self) -> None:
        t = TableTransition(_sprint1_config(), seed=42)
        state = StateView(
            factors={
                "step_bin": "inactive",
                "sleep": "good",
                "day_of_week": "weekday",
                "burden": "low",
            },
            day=0,
            step_of_day=2,
        )
        updates = t.transition(state, "idle")
        assert "sleep" not in updates
        assert "step_bin" in updates


class TestSeedReproducibility:
    def test_same_seed_same_results(self) -> None:
        state = StateView(
            factors={
                "step_bin": "inactive",
                "sleep": "good",
                "day_of_week": "weekday",
                "burden": "low",
            },
            day=0,
            step_of_day=0,
        )
        t1 = TableTransition(_sprint1_config(seed=99), seed=99)
        r1 = t1.transition(state, "idle")

        t2 = TableTransition(_sprint1_config(seed=99), seed=99)
        r2 = t2.transition(state, "idle")
        assert r1 == r2


class TestValidTransitionValues:
    def test_valid_transition_values(self) -> None:
        """All transition outputs must be valid factor values."""
        t = TableTransition(_sprint1_config(), seed=42)
        state = StateView(
            factors={
                "step_bin": "inactive",
                "sleep": "good",
                "day_of_week": "weekday",
                "burden": "low",
            },
            day=0,
            step_of_day=0,
        )
        updates = t.transition(state, "idle")
        assert "sleep" in updates
        assert updates["sleep"] in ("good", "poor")
        assert "step_bin" in updates
        assert updates["step_bin"] in ("inactive", "moderate", "active")


class TestMinimalConfigRejected:
    def test_missing_table_dir_raises(self) -> None:
        """table_transition requires table_dir."""
        with pytest.raises(ValueError, match="table_dir"):
            MDPConfig.model_validate(
                {
                    "episode_days": 1,
                    "steps_per_day": 5,
                    "seed": 42,
                    "state": {"variables": {"x": {"names": ["a", "b"]}}},
                    "initial_state": {"x": "a"},
                    "actions": ["go"],
                    "reward": {
                        "variables": {
                            "v": {"source": "state.x", "mapping": {"a": 0.0, "b": 1.0}}
                        },
                        "formula": "v",
                    },
                    "transition_model": {"type": "table_transition"},
                    "agents": [],
                }
            )

    def test_transition_probabilities_rejected(self) -> None:
        """table_transition rejects transition_probabilities."""
        base = {
            "episode_days": 1,
            "steps_per_day": 5,
            "seed": 42,
            "state": {
                "variables": {
                    "step_bin": {"names": ["inactive", "moderate", "active"]},
                    "sleep": {"names": ["good", "poor"]},
                }
            },
            "initial_state": {"step_bin": "inactive", "sleep": "good"},
            "actions": ["idle"],
            "reward": {
                "variables": {
                    "v": {
                        "source": "state.step_bin",
                        "mapping": {
                            "inactive": 0.0,
                            "moderate": 0.5,
                            "active": 1.0,
                        },
                    }
                },
                "formula": "v",
            },
            "transition_model": {
                "type": "table_transition",
                "table_dir": str(_TABLES_DIR),
                "transition_probabilities": {
                    "inactive": {"idle": {"moderate": 0.5, "inactive": 0.5}},
                    "moderate": {"idle": {"moderate": 0.5, "active": 0.5}},
                    "active": {"idle": {"inactive": 0.5, "active": 0.5}},
                },
            },
            "agents": [],
        }
        with pytest.raises(
            ValueError, match="does not accept transition_probabilities"
        ):
            MDPConfig.model_validate(base)


def _check_all_within_day_keys(t: TableTransition) -> None:
    step_bins = ("inactive", "moderate", "active")
    burdens = ("low", "medium", "high")
    actions = ("idle", "movement_suggestion", "goal_reminder", "journal")
    dows = ("weekday", "weekend")
    sleeps = ("good", "poor")
    for step_idx in range(5):
        wd = t.within_day[step_idx]
        for step_bin, burden, action, dow, sleep in itertools.product(
            step_bins, burdens, actions, dows, sleeps
        ):
            key = f"{step_bin}|{burden}|{action}|{dow}|{sleep}"
            assert key in wd, f"Missing within_day_{step_idx} key: {key}"


class TestWithRealTables:
    """Integration-style tests using the actual JSON tables."""

    def test_tables_are_readable(self) -> None:
        """All 6 table files exist and are valid JSON."""
        assert _TABLES_DIR.exists()
        db_path = _TABLES_DIR / "day_boundary.json"
        assert db_path.exists()
        for i in range(5):
            wd_path = _TABLES_DIR / f"within_day_{i}.json"
            assert wd_path.exists()

    def test_day_boundary_has_all_states(self) -> None:
        t = TableTransition(_sprint1_config(), seed=42)
        for step_bin in ("inactive", "moderate", "active"):
            for burden in ("low", "medium", "high"):
                for dow in ("weekday", "weekend"):
                    for sleep in ("good", "poor"):
                        key = f"{step_bin}|{burden}|{dow}|{sleep}"
                        assert key in t.day_boundary, f"Missing day_boundary key: {key}"

    def test_within_day_has_all_state_action_combos(self) -> None:
        t = TableTransition(_sprint1_config(), seed=42)
        _check_all_within_day_keys(t)


class TestMissingKeyWarning:
    def test_missing_day_boundary_key_logs_warning(self) -> None:
        t = TableTransition(_sprint1_config(), seed=42)
        state = StateView(
            factors={
                "step_bin": "nonexistent",
                "sleep": "good",
                "day_of_week": "weekday",
                "burden": "low",
            },
            day=0,
            step_of_day=0,
        )
        with patch.object(logging.Logger, "warning") as mock_warn:
            t.transition(state, "idle")
            mock_warn.assert_any_call(
                "Missing day_boundary key: %s", "nonexistent|low|weekday|good"
            )

    def test_missing_within_day_key_logs_warning(self) -> None:
        t = TableTransition(_sprint1_config(), seed=42)
        state = StateView(
            factors={
                "step_bin": "nonexistent",
                "sleep": "good",
                "day_of_week": "weekday",
                "burden": "low",
            },
            day=0,
            step_of_day=1,
        )
        with patch.object(logging.Logger, "warning") as mock_warn:
            t.transition(state, "idle")
            mock_warn.assert_any_call(
                "Missing within_day_%d key: %s", 1, "nonexistent|low|idle|weekday|good"
            )


class TestStepThroughDayRange:
    def test_step_of_day_exceeds_range_raises(self) -> None:
        t = TableTransition(_sprint1_config(), seed=42)
        state = StateView(
            factors={
                "step_bin": "inactive",
                "sleep": "good",
                "day_of_week": "weekday",
                "burden": "low",
            },
            day=0,
            step_of_day=99,
        )
        with pytest.raises(
            IndexError, match="step_of_day 99 exceeds within_day table count"
        ):
            t.transition(state, "idle")


@pytest.mark.parametrize(
    ("seed", "action"),
    [
        (0, "idle"),
        (0, "movement_suggestion"),
        (10, "idle"),
        (10, "goal_reminder"),
        (42, "journal"),
    ],
)
def test_valid_transition_parametrized(seed: int, action: str) -> None:
    t = TableTransition(_sprint1_config(), seed=seed)
    state = StateView(
        factors={
            "step_bin": "inactive",
            "sleep": "good",
            "day_of_week": "weekday",
            "burden": "low",
        },
        day=0,
        step_of_day=0,
    )
    updates = t.transition(state, action)
    assert "sleep" in updates
    assert updates["sleep"] in ("good", "poor")
    assert "step_bin" in updates
    assert updates["step_bin"] in ("inactive", "moderate", "active")
