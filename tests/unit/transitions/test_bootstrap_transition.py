from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions.bootstrap import BootstrapTransition


def _sprint_bootstrap_config(table_dir: str, seed: int = 42) -> MDPConfig:
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
            "formula": "alpha * step_bin_value + "
            "(1 - alpha) * sleep_value - action_penalty",
        },
        transition_model={
            "type": "bootstrap",
            "table_dir": table_dir,
        },
        agents=[],
    )


def make_minimal_tables(
    tmp_path: Path,
    *,
    day_boundary_key: str = "inactive,low,weekday,good",
    within_day_key: str = "inactive,low,idle,weekday,good",
) -> Path:
    """Write minimal bootstrap JSON table files into tmp_path and return it.

    The day_boundary table maps to sleep outcomes.
    The within_day tables map to step_bin outcomes.

    Both sleep outcomes (good, poor) are included in within_day tables
    so that a sleep update at step 0 still finds a valid key.
    """
    tables: dict[str, dict[str, dict[str, float]]] = {
        "day_boundary.json": {
            day_boundary_key: {"good": 0.7, "poor": 0.3},
        },
    }
    base_sleep_key = "inactive,low,idle,weekday,"
    for i in range(5):
        tables[f"within_day_{i}.json"] = {
            f"{base_sleep_key}good": {
                "inactive": 0.5,
                "moderate": 0.3,
                "active": 0.2,
            },
            f"{base_sleep_key}poor": {
                "inactive": 0.5,
                "moderate": 0.3,
                "active": 0.2,
            },
        }

    for filename, data in tables.items():
        (tmp_path / filename).write_text(json.dumps(data), encoding="utf-8")

    return tmp_path


class TestTableLoading:
    def test_loads_tables_from_directory(self, tmp_path: Path) -> None:
        table_dir = make_minimal_tables(tmp_path)
        config = _sprint_bootstrap_config(str(table_dir))
        t = BootstrapTransition(config, seed=42)
        assert len(t._tables) == 6
        assert "day_boundary" in t._tables
        for i in range(5):
            assert f"within_day_{i}" in t._tables

    def test_missing_table_raises(self, tmp_path: Path) -> None:
        make_minimal_tables(tmp_path)
        (tmp_path / "within_day_2.json").unlink()
        config = _sprint_bootstrap_config(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="within_day_2"):
            BootstrapTransition(config, seed=42)

    def test_missing_table_dir_raises(self) -> None:
        config = _sprint_bootstrap_config(table_dir="/nonexistent")
        config.transition_model.table_dir = None
        with pytest.raises(ValueError, match="table_dir"):
            BootstrapTransition(config, seed=42)

    def test_validates_probabilities_sum_to_one(self, tmp_path: Path) -> None:
        table_dir = make_minimal_tables(tmp_path)
        bad = {"inactive,low,weekday,good": {"good": 0.5, "poor": 0.4}}
        (table_dir / "day_boundary.json").write_text(json.dumps(bad), encoding="utf-8")
        config = _sprint_bootstrap_config(str(table_dir))
        with pytest.raises(ValueError, match=r"sum to 0.9"):
            BootstrapTransition(config, seed=42)


class TestKeyBuilding:
    def test_day_boundary_key(self) -> None:
        state = StateView(
            factors={
                "step_bin": "inactive",
                "step_bin_daily": "active",
                "sleep": "good",
                "day_of_week": "weekday",
                "burden": "low",
            },
            day=0,
            step_of_day=0,
        )
        config = _sprint_bootstrap_config("/dummy")
        t = BootstrapTransition.__new__(BootstrapTransition)
        t._config = config  # type: ignore[attr-defined]
        key = t._build_key(state, "idle", for_day_boundary=True)
        assert key == "active,low,weekday,good"

    def test_within_day_key(self) -> None:
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
        config = _sprint_bootstrap_config("/dummy")
        t = BootstrapTransition.__new__(BootstrapTransition)
        t._config = config  # type: ignore[attr-defined]
        key = t._build_key(state, "idle")
        assert key == "inactive,low,idle,weekday,good"


class TestTransition:
    def test_step_zero_samples_sleep_and_step_bin(self, tmp_path: Path) -> None:
        table_dir = make_minimal_tables(tmp_path)
        config = _sprint_bootstrap_config(str(table_dir))
        t = BootstrapTransition(config, seed=42)
        state = StateView(
            factors={
                "step_bin": "inactive",
                "step_bin_daily": "inactive",
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

    def test_step_nonzero_samples_only_step_bin(self, tmp_path: Path) -> None:
        table_dir = make_minimal_tables(tmp_path)
        config = _sprint_bootstrap_config(str(table_dir))
        t = BootstrapTransition(config, seed=42)
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

    def test_step_zero_missing_step_bin_daily_raises(self, tmp_path: Path) -> None:
        table_dir = make_minimal_tables(tmp_path)
        config = _sprint_bootstrap_config(str(table_dir))
        t = BootstrapTransition(config, seed=42)
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
        with pytest.raises(ValueError, match="step_bin_daily"):
            t.transition(state, "idle")


class TestSeedReproducibility:
    def test_same_seed_same_results(self, tmp_path: Path) -> None:
        table_dir = make_minimal_tables(tmp_path)
        config = _sprint_bootstrap_config(str(table_dir), seed=42)
        state = StateView(
            factors={
                "step_bin": "inactive",
                "step_bin_daily": "inactive",
                "sleep": "good",
                "day_of_week": "weekday",
                "burden": "low",
            },
            day=0,
            step_of_day=0,
        )
        t1 = BootstrapTransition(config, seed=99)
        r1 = t1.transition(state, "idle")
        t2 = BootstrapTransition(config, seed=99)
        r2 = t2.transition(state, "idle")
        assert r1 == r2

    def test_different_seed_different_results(self, tmp_path: Path) -> None:
        table_dir = make_minimal_tables(tmp_path)
        config = _sprint_bootstrap_config(str(table_dir), seed=42)
        state = StateView(
            factors={
                "step_bin": "inactive",
                "step_bin_daily": "inactive",
                "sleep": "good",
                "day_of_week": "weekday",
                "burden": "low",
            },
            day=0,
            step_of_day=0,
        )
        results: set[tuple[str | None, str | None]] = set()
        for s in range(20):
            t = BootstrapTransition(config, seed=s)
            r = t.transition(state, "idle")
            results.add((r.get("step_bin"), r.get("sleep")))
        assert len(results) > 1


class TestDistribution:
    def test_probability_distribution(self, tmp_path: Path) -> None:
        """Run many samples and verify empirical dist matches table probs."""
        table_dir = make_minimal_tables(tmp_path)
        config = _sprint_bootstrap_config(str(table_dir), seed=42)
        t = BootstrapTransition(config, seed=0)
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
        n_samples = 10_000
        counts: dict[str, int] = {"inactive": 0, "moderate": 0, "active": 0}
        for _ in range(n_samples):
            updates = t.transition(state, "idle")
            counts[updates["step_bin"]] += 1

        expected = {"inactive": 0.5, "moderate": 0.3, "active": 0.2}
        for outcome, prob in expected.items():
            observed = counts[outcome] / n_samples
            assert abs(observed - prob) < 0.025, (
                f"Outcome '{outcome}': expected {prob:.3f}, observed {observed:.3f}"
            )


class TestTableIndexClamping:
    def test_step_index_clamped_to_four(self, tmp_path: Path) -> None:
        """step_of_day > 4 should still use within_day_4 table."""
        table_dir = make_minimal_tables(tmp_path)
        config = _sprint_bootstrap_config(str(table_dir), seed=42)
        t = BootstrapTransition(config, seed=42)
        state = StateView(
            factors={
                "step_bin": "inactive",
                "sleep": "good",
                "day_of_week": "weekday",
                "burden": "low",
            },
            day=0,
            step_of_day=10,
        )
        updates = t.transition(state, "idle")
        assert "step_bin" in updates
        assert updates["step_bin"] in ("inactive", "moderate", "active")


class TestRegistryIntegration:
    def test_bootstrap_in_registry(self) -> None:
        from rl_health_interventions.transitions import REGISTRY

        assert "bootstrap" in REGISTRY
        assert REGISTRY["bootstrap"] is BootstrapTransition


class TestLoadTablesEdgeCases:
    def test_missing_key_raises_key_error(self, tmp_path: Path) -> None:
        """Requesting a key not in the table raises KeyError."""
        table_dir = make_minimal_tables(
            tmp_path,
            day_boundary_key="inactive,low,weekday,good",
            within_day_key="inactive,low,idle,weekday,good",
        )
        config = _sprint_bootstrap_config(str(table_dir), seed=42)
        t = BootstrapTransition(config, seed=42)
        state = StateView(
            factors={
                "step_bin": "active",
                "sleep": "poor",
                "day_of_week": "weekend",
                "burden": "high",
            },
            day=0,
            step_of_day=2,
        )
        with pytest.raises(KeyError):
            t.transition(state, "movement_suggestion")
