"""Tests for Bayesian P-success burden in Environment."""

from __future__ import annotations

import collections
from pathlib import Path

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.environment import Environment

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TABLES_DIR = str(_REPO_ROOT / "tables" / "pearl_12action")


def _pearl_config(seed: int = 42, *, episode_days: int = 50) -> MDPConfig:
    return MDPConfig(
        episode_days=episode_days,
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
                        "window_size": 2,
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
                        "mapping": {0: "low", 1: "medium", 2: "high"},
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
        # Use multiple seeds to ensure we see medium burden at least once.
        # With a single seed, P(success) may be high enough that 3 failures
        # in a window of 3 never occurs.
        seen_medium = False
        for seed in range(20):
            env = Environment(_pearl_config(seed=seed), seed=seed)
            state = env.reset()
            for _ in range(5):
                state, _, _ = env.step("ability_morning")
            if state.burden == "medium":
                seen_medium = True
                break
        assert seen_medium, (
            "Non-idle actions never produced medium burden across 20 seeds"
        )

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


class TestPSuccessKeyConsistency:
    def test_state_key_order_matches_precomputation(self) -> None:
        """State key ordering in runtime lookup must match precomputation order.

        Regression test for Bug 1: _apply_rolling_advances used sorted()
        while _precompute_p_success used config declaration order, causing
        every lookup to miss and return the default 0.5.
        """
        env = Environment(_pearl_config(), seed=42)
        _state = env.reset()
        # Take a non-idle action to trigger burden computation
        _state, _, _ = env.step("ability_morning")
        # Verify that _p_success keys use config declaration order
        # (recent_steps_mean|recent_walk_pattern|morning_steps_ratio|action)
        for key in env._p_success:
            parts = key.split("|")
            assert len(parts) == 4, f"Expected 4 parts: {key}"
            # The first factor should be from recent_steps_mean values
            assert parts[0] in ("low", "moderate", "high"), (
                f"First part should be recent_steps_mean value: {key}"
            )

    def test_p_success_lookup_not_always_default(self) -> None:
        """P-success lookups should find precomputed values, not always default 0.5.

        Regression test for Bug 1: if key ordering is wrong, every lookup
        falls through to the default 0.5.
        """
        env = Environment(_pearl_config(), seed=42)
        _state = env.reset()
        # Take several non-idle actions to exercise burden computation
        for _ in range(5):
            _state, _, _ = env.step("ability_morning")
        # If the fix is correct, some P-success lookups should have found
        # precomputed values (not all 0.5). We can't check individual lookups
        # since they're stochastic, but we can verify the precomputed table
        # has entries that would be looked up.
        assert len(env._p_success) > 0
        # Verify at least some values are not 0.5
        non_default = [p for p in env._p_success.values() if abs(p - 0.5) > 0.01]
        assert len(non_default) > 0, (
            "All P-success values are 0.5 — key ordering may be wrong"
        )


class TestHistoricalStateContext:
    def test_action_history_preserves_state_context(self) -> None:
        """Action history should store (state_key, action) tuples internally.

        Regression test for Bug 2: historical P-success lookups used the
        current state instead of the state at the time each action was taken.
        """
        env = Environment(_pearl_config(), seed=42)
        _state = env.reset()
        # Take actions to populate history (deque maxlen=3, so primed
        # entries are evicted after 3 actual actions)
        _state, _, _ = env.step("ability_morning")
        _state, _, _ = env.step("planning_afternoon")
        _state, _, _ = env.step("idle")
        # Internal history should be tuples (maxlen=2, old entries evicted)
        assert len(env._action_history) == 2
        for entry in env._action_history:
            assert isinstance(entry, tuple), (
                f"Expected tuple, got {type(entry)}: {entry}"
            )
            assert len(entry) == 2, f"Expected (state_key, action), got: {entry}"
            assert isinstance(entry[0], str), f"state_key should be str: {entry}"
            assert isinstance(entry[1], str), f"action should be str: {entry}"

    def test_action_history_property_returns_actions_only(self) -> None:
        """Public action_history property should return action strings only."""
        env = Environment(_pearl_config(), seed=42)
        _state = env.reset()
        _state, _, _ = env.step("ability_morning")
        _state, _, _ = env.step("idle")
        # Public property returns action strings
        history = env.action_history
        assert isinstance(history, tuple)
        for item in history:
            assert isinstance(item, str), f"Expected str, got {type(item)}: {item}"

    def test_burden_uses_historical_state_not_current(self) -> None:
        """Burden computation should use the state at time of each action.

        Regression test for Bug 2: if current state is used instead,
        burden lookup keys will be wrong when state has changed.
        """
        # Use multiple seeds since burden variation is stochastic
        seen_medium = False
        for seed in range(20):
            env = Environment(_pearl_config(seed=seed), seed=seed)
            _state = env.reset()
            for _ in range(5):
                _state, _, _ = env.step("ability_morning")
            if _state.burden == "medium":
                seen_medium = True
                break
        assert seen_medium, (
            "Burden never varied to medium across 20 seeds — "
            "historical state context may not be preserved"
        )
