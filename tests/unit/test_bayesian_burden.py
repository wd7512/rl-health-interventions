"""Tests for Bayesian P-success burden in Environment."""

from __future__ import annotations

import collections
from pathlib import Path

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.environment import Environment

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TABLES_DIR = str(_REPO_ROOT / "tables" / "pearl_12action")


def _pearl_config(seed: int = 42) -> MDPConfig:
    return MDPConfig(
        episode_days=15,
        steps_per_day=1,
        seed=seed,
        state={
            "variables": {
                "recent_steps_mean": {"names": ["low", "moderate", "high"]},
                "recent_walk_pattern": {"names": ["low", "high"]},
                "morning_steps_ratio": {"names": ["morning", "balanced", "evening"]},
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
                                    "ability_morning",
                                    "planning_afternoon",
                                ],
                            }
                        ],
                        "mapping": {0: "low", 1: "low", 2: "low", 3: "medium"},
                    },
                },
            }
        },
        initial_state={
            "recent_steps_mean": "moderate",
            "recent_walk_pattern": "low",
            "morning_steps_ratio": "balanced",
            "burden": "low",
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
        transition_model={"type": "table_transition", "table_dir": _TABLES_DIR},
        agents=[],
    )


class TestPrecomputePSuccess:
    def test_p_success_populated(self) -> None:
        env = Environment(_pearl_config(), seed=42)
        assert len(env._p_success) > 0

    def test_p_success_range(self) -> None:
        env = Environment(_pearl_config(), seed=42)
        for key, p in env._p_success.items():
            assert 0.0 <= p <= 1.0, f"{key}: p_success={p} out of range"

    def test_idle_not_in_p_success(self) -> None:
        env = Environment(_pearl_config(), seed=42)
        for key in env._p_success:
            assert not key.endswith("|idle"), f"Idle should not be in p_success: {key}"

    def test_p_success_key_format(self) -> None:
        env = Environment(_pearl_config(), seed=42)
        for key in env._p_success:
            parts = key.split("|")
            assert len(parts) == 4, f"Expected 4 parts (3 factors + action): {key}"
            assert parts[-1] != "idle"


class TestBurdenVariation:
    def test_idle_always_low_burden(self) -> None:
        env = Environment(_pearl_config(), seed=42)
        state = env.reset()
        for _ in range(10):
            state, _, _ = env.step("idle")
        assert state.burden == "low"

    def test_nonidle_actions_vary_burden(self) -> None:
        env = Environment(_pearl_config(), seed=42)
        state = env.reset()
        burdens = []
        for _ in range(10):
            state, _, _ = env.step("ability_morning")
            burdens.append(state.burden)
        counts = collections.Counter(burdens)
        # Non-idle actions should produce some medium burden
        assert counts.get("medium", 0) > 0

    def test_different_actions_different_burden_profiles(self) -> None:
        burdens_idle = []
        burdens_action = []
        for seed in range(20):
            env = Environment(_pearl_config(seed=seed), seed=seed)
            state = env.reset()
            for _ in range(5):
                state, _, _ = env.step("idle")
            burdens_idle.append(state.burden)

            env2 = Environment(_pearl_config(seed=seed), seed=seed)
            state2 = env2.reset()
            for _ in range(5):
                state2, _, _ = env2.step("ability_morning")
            burdens_action.append(state2.burden)

        idle_counts = collections.Counter(burdens_idle)
        action_counts = collections.Counter(burdens_action)
        # Idle should be all low
        assert idle_counts.get("low", 0) == 20
        # Action should have some medium
        assert action_counts.get("medium", 0) > 0


class TestFallbackBehavior:
    def test_random_transition_no_p_success(self) -> None:
        """RandomTransition (no tables) should not precompute P-success."""
        config = MDPConfig(
            episode_days=1,
            steps_per_day=1,
            seed=42,
            state={
                "variables": {
                    "activity_level": {"names": ["sedentary", "active"]},
                    "burden": {
                        "names": ["low", "medium", "high"],
                        "advanced": {
                            "type": "rolling_window_count",
                            "window_size": 3,
                            "conditions": [
                                {"factor": "action", "type": "in", "values": ["nudge"]}
                            ],
                            "mapping": {0: "low", 1: "low", 2: "low", 3: "medium"},
                        },
                    },
                }
            },
            initial_state={"activity_level": "sedentary", "burden": "low"},
            actions=["idle", "nudge"],
            reward={
                "variables": {
                    "v": {
                        "source": "state.activity_level",
                        "mapping": {"sedentary": 0.0, "active": 1.0},
                    }
                },
                "formula": "v",
            },
            transition_model={"type": "random"},
            agents=[],
        )
        env = Environment(config, seed=42)
        assert len(env._p_success) == 0
