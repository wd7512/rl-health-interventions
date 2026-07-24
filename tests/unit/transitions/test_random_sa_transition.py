"""Unit tests for RandomTransitionSA transition model."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions.random_sa import RandomTransitionSA


def _pearl_config(seed: int = 42) -> MDPConfig:
    return MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=seed,
        state={
            "variables": {
                "recent_steps_mean": {"names": ["low", "moderate", "high"]},
                "recent_walk_pattern": {"names": ["low", "high"]},
                "morning_steps_ratio": {"names": ["morning", "balanced", "evening"]},
            }
        },
        initial_state={
            "recent_steps_mean": "moderate",
            "recent_walk_pattern": "low",
            "morning_steps_ratio": "balanced",
        },
        actions=["idle", "ability_morning", "planning_afternoon"],
        reward={
            "variables": {
                "v": {
                    "source": "state.recent_steps_mean",
                    "mapping": {"low": -1.0, "moderate": 0.0, "high": 1.0},
                }
            },
            "formula": "v",
        },
        transition_model={"type": "random_sa"},
        agents=[],
    )


class TestConstruction:
    def test_creates_tables(self) -> None:
        t = RandomTransitionSA(_pearl_config(), seed=42)
        assert len(t.day_boundary) > 0
        assert len(t.within_day) == 1
        assert len(t.within_day[0]) > 0

    def test_day_boundary_key_format(self) -> None:
        t = RandomTransitionSA(_pearl_config(), seed=42)
        for key in t.day_boundary:
            parts = key.split("|")
            assert len(parts) == 3  # 3 stochastic factors

    def test_within_day_key_format(self) -> None:
        t = RandomTransitionSA(_pearl_config(), seed=42)
        for key in t.within_day[0]:
            parts = key.split("|")
            assert len(parts) == 4  # 3 factors + action


class TestTableStructure:
    def test_day_boundary_probs_sum_to_one(self) -> None:
        t = RandomTransitionSA(_pearl_config(), seed=42)
        for key, (targets, probs) in t.day_boundary.items():
            assert len(targets) == len(probs)
            assert abs(float(probs.sum()) - 1.0) < 1e-6
            assert np.all(probs > 0), f"{key}: has non-positive probability"

    def test_within_day_probs_sum_to_one(self) -> None:
        t = RandomTransitionSA(_pearl_config(), seed=42)
        for key, (targets, probs) in t.within_day[0].items():
            assert len(targets) == len(probs)
            assert abs(float(probs.sum()) - 1.0) < 1e-6
            assert np.all(probs > 0), f"{key}: has non-positive probability"


class TestTransition:
    def test_returns_valid_values(self) -> None:
        t = RandomTransitionSA(_pearl_config(), seed=42)
        state = StateView(
            factors={
                "recent_steps_mean": "moderate",
                "recent_walk_pattern": "low",
                "morning_steps_ratio": "balanced",
            },
            day=0,
            step_of_day=0,
        )
        updates = t.transition(state, "ability_morning")
        assert "recent_steps_mean" in updates
        assert "recent_walk_pattern" in updates
        assert "morning_steps_ratio" in updates
        assert updates["recent_steps_mean"] in ("low", "moderate", "high")
        assert updates["recent_walk_pattern"] in ("low", "high")
        assert updates["morning_steps_ratio"] in ("morning", "balanced", "evening")

    def test_different_actions_give_different_distributions(self) -> None:
        counts_idle: dict[str, int] = {}
        counts_action: dict[str, int] = {}
        for seed in range(50):
            t = RandomTransitionSA(_pearl_config(seed=seed), seed=seed)
            state = StateView(
                factors={
                    "recent_steps_mean": "low",
                    "recent_walk_pattern": "low",
                    "morning_steps_ratio": "morning",
                },
                day=0,
                step_of_day=0,
            )
            u_idle = t.transition(state, "idle")
            u_action = t.transition(state, "ability_morning")
            k = u_idle["recent_steps_mean"]
            counts_idle[k] = counts_idle.get(k, 0) + 1
            k = u_action["recent_steps_mean"]
            counts_action[k] = counts_action.get(k, 0) + 1
        # Both should produce some variation (not all same value)
        assert len(counts_idle) > 1
        assert len(counts_action) > 1


class TestSeedReproducibility:
    def test_same_seed_same_results(self) -> None:
        state = StateView(
            factors={
                "recent_steps_mean": "moderate",
                "recent_walk_pattern": "low",
                "morning_steps_ratio": "balanced",
            },
            day=0,
            step_of_day=0,
        )
        t1 = RandomTransitionSA(_pearl_config(seed=99), seed=99)
        r1 = t1.transition(state, "idle")
        t2 = RandomTransitionSA(_pearl_config(seed=99), seed=99)
        r2 = t2.transition(state, "idle")
        assert r1 == r2


class TestSaveTables:
    def test_save_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            t = RandomTransitionSA(_pearl_config(), seed=42)
            t.save_tables(tmpdir)
            db_path = Path(tmpdir) / "day_boundary.json"
            wd_path = Path(tmpdir) / "within_day_0.json"
            assert db_path.exists()
            assert wd_path.exists()
            with db_path.open(encoding="utf-8") as f:
                db_data = json.load(f)
            with wd_path.open(encoding="utf-8") as f:
                wd_data = json.load(f)
            assert len(db_data) > 0
            assert len(wd_data) > 0
            # Check structure
            for _key, entry in db_data.items():
                assert "_" in entry
                probs = list(entry["_"].values())
                assert abs(sum(probs) - 1.0) < 1e-6

    def test_saved_tables_loadable_by_bootstrap(self) -> None:
        from rl_health_interventions.transitions.bootstrap import BootstrapTransition

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _pearl_config()
            t = RandomTransitionSA(config, seed=42)
            t.save_tables(tmpdir)
            # Update config to point to saved tables
            config.transition_model.table_dir = tmpdir
            bt = BootstrapTransition(config, seed=42)
            assert len(bt.day_boundary) > 0
            assert len(bt.within_day) == 1


class TestMissingKey:
    def test_missing_key_logs_warning(self) -> None:
        import logging
        from unittest.mock import patch

        t = RandomTransitionSA(_pearl_config(), seed=42)
        state = StateView(
            factors={
                "recent_steps_mean": "nonexistent",
                "recent_walk_pattern": "low",
                "morning_steps_ratio": "balanced",
            },
            day=0,
            step_of_day=0,
        )
        with patch.object(logging.Logger, "warning") as mock_warn:
            t.transition(state, "idle")
            mock_warn.assert_any_call(
                "Missing day_boundary key for factor %s value %s",
                "recent_steps_mean",
                "nonexistent",
            )
