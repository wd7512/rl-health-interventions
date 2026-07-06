from __future__ import annotations

import numpy as np

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions.random import RandomTransition


def _sprint1_random_config(seed: int = 42) -> MDPConfig:
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
        transition_model={"type": "random"},
        agents=[],
    )


class TestDirichletValidity:
    def test_probabilities_sum_to_one(self) -> None:
        t = RandomTransition(_sprint1_random_config(seed=0), seed=0)
        for name, (targets, probs) in t._cache.items():
            assert len(probs) == len(targets), f"{name}: length mismatch"
            assert abs(float(probs.sum()) - 1.0) < 1e-6, f"{name}: don't sum to 1"

    def test_all_probabilities_positive(self) -> None:
        t = RandomTransition(_sprint1_random_config(seed=0), seed=0)
        for name, (_, probs) in t._cache.items():
            assert np.all(probs > 0), f"{name}: has non-positive probability"

    def test_cache_keys_match_stochastic_factors(self) -> None:
        t = RandomTransition(_sprint1_random_config(seed=0), seed=0)
        assert set(t._cache.keys()) == set(t._stochastic_factors)


class TestSeedReproducibility:
    def test_same_seed_same_results(self) -> None:
        config = _sprint1_random_config(seed=42)
        state = StateView(
            factors={"step_bin": "inactive", "sleep": "good"}, day=0, step_of_day=0
        )
        t1 = RandomTransition(config, seed=99)
        r1 = t1.transition(state, "idle")
        t2 = RandomTransition(config, seed=99)
        r2 = t2.transition(state, "idle")
        assert r1 == r2

    def test_different_seed_different_results(self) -> None:
        config = _sprint1_random_config(seed=42)
        state = StateView(
            factors={"step_bin": "inactive", "sleep": "good"}, day=0, step_of_day=0
        )
        results = set()
        for s in range(20):
            t = RandomTransition(config, seed=s)
            r = t.transition(state, "idle")
            results.add((r.get("step_bin"), r.get("sleep")))
        assert len(results) > 1


class TestRouting:
    def test_step_zero_samples_both_sleep_and_step_bin(self) -> None:
        t = RandomTransition(_sprint1_random_config(seed=42), seed=42)
        state = StateView(
            factors={"step_bin": "inactive", "sleep": "good"}, day=0, step_of_day=0
        )
        updates = t.transition(state, "idle")
        assert "sleep" in updates
        assert "step_bin" in updates
        assert updates["sleep"] in ("good", "poor")
        assert updates["step_bin"] in ("inactive", "moderate", "active")

    def test_step_nonzero_samples_only_step_bin(self) -> None:
        t = RandomTransition(_sprint1_random_config(seed=42), seed=42)
        state = StateView(
            factors={"step_bin": "inactive", "sleep": "good"}, day=0, step_of_day=2
        )
        updates = t.transition(state, "idle")
        assert "sleep" not in updates
        assert "step_bin" in updates

    def test_returns_dict(self) -> None:
        t = RandomTransition(_sprint1_random_config(seed=42), seed=42)
        state = StateView(
            factors={"step_bin": "inactive", "sleep": "good"}, day=0, step_of_day=0
        )
        result = t.transition(state, "idle")
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, str)


class TestStochasticFactors:
    def test_only_non_advanced_factors(self) -> None:
        t = RandomTransition(_sprint1_random_config(seed=42), seed=42)
        assert "sleep" in t._stochastic_factors
        assert "step_bin" in t._stochastic_factors
        assert "day_of_week" not in t._stochastic_factors
        assert "burden" not in t._stochastic_factors


class TestMinimalConfig:
    def test_single_stochastic_factor(self) -> None:
        config = MDPConfig(
            episode_days=1,
            steps_per_day=1,
            seed=42,
            state={"variables": {"x": {"names": ["a", "b"]}}},
            initial_state={"x": "a"},
            actions=["go"],
            reward={
                "variables": {
                    "v": {"source": "state.x", "mapping": {"a": 0.0, "b": 1.0}}
                },
                "formula": "v",
            },
            transition_model={"type": "random"},
            agents=[],
        )
        t = RandomTransition(config, seed=42)
        assert set(t._cache.keys()) == {"x"}
        state = StateView(factors={"x": "a"}, day=0, step_of_day=0)
        updates = t.transition(state, "go")
        assert "x" in updates
        assert updates["x"] in ("a", "b")

    def test_no_stochastic_factors(self) -> None:
        config = MDPConfig(
            episode_days=1,
            steps_per_day=1,
            seed=42,
            state={
                "variables": {
                    "weekday_flag": {
                        "names": ["d1", "d2"],
                        "advanced": {
                            "type": "cyclic",
                            "granularity": "daily",
                            "pattern": ["d1", "d2"],
                        },
                    }
                }
            },
            initial_state={"weekday_flag": "d1"},
            actions=["go"],
            reward={
                "variables": {
                    "v": {
                        "source": "state.weekday_flag",
                        "mapping": {"d1": 0.0, "d2": 1.0},
                    }
                },
                "formula": "v",
            },
            transition_model={"type": "random"},
            agents=[],
        )
        t = RandomTransition(config, seed=42)
        assert t._cache == {}
        state = StateView(factors={"weekday_flag": "d1"}, day=0, step_of_day=0)
        updates = t.transition(state, "go")
        assert updates == {}
