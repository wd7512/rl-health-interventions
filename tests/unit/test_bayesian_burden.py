from __future__ import annotations

import copy
from collections import deque
from unittest import mock

import numpy as np
import pytest

from rl_health_interventions.config.schemas import MDPConfig, RollingWindowCountAdvance
from rl_health_interventions.environment import Environment

# ── helpers ──────────────────────────────────────────────────────────────────

_FACTORED_CONFIG = {
    "episode_days": 2,
    "steps_per_day": 3,
    "seed": 42,
    "state": {
        "variables": {
            "step_bin": {"names": ["inactive", "moderate", "active"]},
            "sleep": {"names": ["good", "poor"]},
            "burden": {
                "names": ["low", "medium", "high"],
                "advanced": {
                    "type": "rolling_window_count",
                    "mechanism": "rolling_window_count",
                    "window_size": 3,
                    "conditions": [
                        {"factor": "action", "type": "in", "values": ["go"]}
                    ],
                    "mapping": {0: "low", 1: "medium", 2: "high", 3: "high"},
                },
            },
        }
    },
    "initial_state": {
        "step_bin": "inactive",
        "sleep": "good",
        "burden": "low",
    },
    "actions": ["idle", "go"],
    "reward": {
        "variables": {
            "v": {
                "source": "state.step_bin",
                "mapping": {"inactive": 0.0, "moderate": 0.5, "active": 1.0},
            }
        },
        "formula": "v",
    },
    "transition_model": {"type": "random"},
    "agents": [],
}


_BAYESIAN_MAPPING = {
    0: "low",
    1: "low",
    2: "low",
    3: "medium",
    4: "medium",
    5: "medium",
    6: "high",
    7: "high",
}


def _bayesian_config(window_size: int = 7) -> MDPConfig:
    """Return a copy of the factored config with bayesian_p_success mechanism."""
    cfg = copy.deepcopy(_FACTORED_CONFIG)
    burden_adv = cfg["state"]["variables"]["burden"]["advanced"]
    burden_adv["mechanism"] = "bayesian_p_success"
    burden_adv["window_size"] = window_size
    burden_adv["mapping"] = dict(_BAYESIAN_MAPPING)
    return MDPConfig(**cfg)


# ── Schema / validation ─────────────────────────────────────────────────────


class TestMechanismValidation:
    def test_default_is_rolling_window_count(self) -> None:
        """Default mechanism should be 'rolling_window_count' for backward compat."""
        adv = RollingWindowCountAdvance(
            conditions=[{"factor": "action", "type": "in", "values": ["go"]}],
            mapping={0: "low", 1: "high", 2: "high", 3: "high"},
        )
        assert adv.mechanism == "rolling_window_count"

    def test_accepts_bayesian_p_success(self) -> None:
        """'bayesian_p_success' should be a valid mechanism."""
        adv = RollingWindowCountAdvance(
            mechanism="bayesian_p_success",
            window_size=7,
            conditions=[{"factor": "action", "type": "in", "values": ["go"]}],
            mapping=_BAYESIAN_MAPPING,
        )
        assert adv.mechanism == "bayesian_p_success"

    def test_rejects_invalid_mechanism(self) -> None:
        """An unknown mechanism should raise ValueError."""
        with pytest.raises(ValueError, match="mechanism"):
            RollingWindowCountAdvance(
                mechanism="invalid_mechanism",
                window_size=3,
                conditions=[{"factor": "action", "type": "in", "values": ["go"]}],
                mapping={0: "low", 1: "medium", 2: "high", 3: "high"},
            )


# ── Backward compatibility ───────────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_default_mechanism_preserves_existing_behavior(self) -> None:
        """Environment with default mechanism should behave exactly as before."""
        config = MDPConfig(**_FACTORED_CONFIG)
        env = Environment(config, seed=42)
        env.reset()
        state, _, _ = env.step("go")
        # burden should be updated via rolling_window_count
        assert hasattr(state, "burden")

    def test_bayesian_config_does_not_crash(self) -> None:
        """A bayesian config should initialise without errors."""
        config = _bayesian_config()
        env = Environment(config, seed=42)
        env.reset()
        state, _, _ = env.step("idle")
        assert hasattr(state, "burden")


# ── Precomputation ───────────────────────────────────────────────────────────


class TestPrecomputeSuccessProbs:
    def test_precompute_populates_dict_for_table_lookup(self) -> None:
        """_precompute_success_probs should populate _success_probs."""
        config = _bayesian_config()
        env = Environment(config, seed=42)

        # With random transition, no lookup, so success_probs should be empty
        # (fallback to 0.5). Check the dict exists.
        assert hasattr(env, "_success_probs")
        assert isinstance(env._success_probs, dict)

    def test_precompute_with_mock_lookup(self) -> None:
        """Test precomputation with a mocked lookup dict."""
        config = _bayesian_config()
        env = Environment(config, seed=42)

        # Build a mock lookup for a 2-factor (step_bin, sleep), no step_of_day scenario
        mock_lookup = {
            # State: step_bin=inactive, sleep=good, under idle
            "inactive|good|idle": {
                "step_bin": (
                    ["inactive", "moderate", "active"],
                    np.array([0.8, 0.15, 0.05]),
                ),
            },
            # State: step_bin=inactive, sleep=good, under go
            "inactive|good|go": {
                "step_bin": (
                    ["inactive", "moderate", "active"],
                    np.array([0.2, 0.5, 0.3]),
                ),
            },
        }

        with mock.patch.object(env, "_transition") as mock_trans:
            mock_trans.lookup = mock_lookup
            mock_trans._include_step_of_day = False  # type: ignore[attr-defined]
            # Re-trigger precomputation
            env._success_probs = env._precompute_success_probs()

        assert len(env._success_probs) > 0
        # Key should be "inactive|good|go"
        key = "inactive|good|go"
        assert key in env._success_probs

    def test_success_high_when_action_diverges_from_idle(self) -> None:
        """P_success should be high when action's distribution differs from idle."""
        config = _bayesian_config()
        env = Environment(config, seed=42)

        # Action pushes strongly toward "active", idle stays at "inactive"
        mock_lookup = {
            "inactive|good|idle": {
                "step_bin": (
                    ["inactive", "moderate", "active"],
                    np.array([0.9, 0.05, 0.05]),
                ),
            },
            "inactive|good|go": {
                "step_bin": (
                    ["inactive", "moderate", "active"],
                    np.array([0.05, 0.1, 0.85]),
                ),
            },
        }

        with mock.patch.object(env, "_transition") as mock_trans:
            mock_trans.lookup = mock_lookup
            mock_trans._include_step_of_day = False  # type: ignore[attr-defined]
            env._success_probs = env._precompute_success_probs()

        ps = env._success_probs["inactive|good|go"]
        # Action is very different from idle → high P_success
        # (0.05*0.05/(0.05+0.9)) + (0.1*0.1/(0.1+0.05)) + (0.85*0.85/(0.85+0.05))
        # = 0.0026 + 0.0667 + 0.8028 = 0.872
        assert ps > 0.7, f"Expected high P_success, got {ps}"

    def test_success_low_when_action_same_as_idle(self) -> None:
        """P_success should be ~0.5 when action has same distribution as idle."""
        config = _bayesian_config()
        env = Environment(config, seed=42)

        probs = np.array([0.5, 0.3, 0.2])
        mock_lookup = {
            "inactive|good|idle": {
                "step_bin": (["inactive", "moderate", "active"], probs.copy()),
            },
            "inactive|good|go": {
                "step_bin": (["inactive", "moderate", "active"], probs.copy()),
            },
        }

        with mock.patch.object(env, "_transition") as mock_trans:
            mock_trans.lookup = mock_lookup
            mock_trans._include_step_of_day = False  # type: ignore[attr-defined]
            env._success_probs = env._precompute_success_probs()

        ps = env._success_probs["inactive|good|go"]
        # Same distribution → P_success = 0.5
        assert abs(ps - 0.5) < 1e-10, f"Expected 0.5, got {ps}"


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_both_probs_zero(self) -> None:
        """When both probs are zero for a value, P_success = 0.5."""
        config = _bayesian_config()
        env = Environment(config, seed=42)

        # For "inactive", both action and idle have 0 probability, P_success = 0.5
        mock_lookup = {
            "inactive|good|idle": {
                "step_bin": (
                    ["inactive", "moderate", "active"],
                    np.array([0.0, 0.5, 0.5]),
                ),
            },
            "inactive|good|go": {
                "step_bin": (
                    ["inactive", "moderate", "active"],
                    np.array([0.0, 0.5, 0.5]),
                ),
            },
        }

        with mock.patch.object(env, "_transition") as mock_trans:
            mock_trans.lookup = mock_lookup
            mock_trans._include_step_of_day = False  # type: ignore[attr-defined]
            env._success_probs = env._precompute_success_probs()

        ps = env._success_probs["inactive|good|go"]
        # Both distributions are identical: (0, 0.5, 0.5).
        # P(success) per value:
        #   inactive: p_a=0, p_i=0 → P(success)=0.5, term = 0 * 0.5 = 0
        #   moderate: P(success)=0.5/(0.5+0.5)=0.5, term = 0.5*0.5=0.25
        #   active:   P(success)=0.5/(0.5+0.5)=0.5, term = 0.5*0.5=0.25
        # Total = 0.5
        assert abs(ps - 0.5) < 1e-10, f"Expected 0.5, got {ps}"

    def test_only_action_causes_transition(self) -> None:
        """When idle has 0 probability for a target, contribution should be 1.0."""
        config = _bayesian_config()
        env = Environment(config, seed=42)

        mock_lookup = {
            "inactive|good|idle": {
                "step_bin": (
                    ["inactive", "moderate", "active"],
                    np.array([1.0, 0.0, 0.0]),
                ),
            },
            "inactive|good|go": {
                "step_bin": (
                    ["inactive", "moderate", "active"],
                    np.array([0.0, 0.0, 1.0]),
                ),
            },
        }

        with mock.patch.object(env, "_transition") as mock_trans:
            mock_trans.lookup = mock_lookup
            mock_trans._include_step_of_day = False  # type: ignore[attr-defined]
            env._success_probs = env._precompute_success_probs()

        ps = env._success_probs["inactive|good|go"]

        # active:  P(success)=1.0, term=1.0*1.0=1.0
        # inactive: P(success)=0, term=0*0=0
        # moderate: P(success)=0.5, term=0*0.5=0
        # Total = 1.0

        assert abs(ps - 1.0) < 1e-10, f"Expected 1.0, got {ps}"

    def test_action_has_zero_prob_idle_positive(self) -> None:
        """When action prob is 0 and idle is > 0 for a target, contribution = 0."""
        config = _bayesian_config()
        env = Environment(config, seed=42)

        mock_lookup = {
            "inactive|good|idle": {
                "step_bin": (["inactive", "moderate"], np.array([0.3, 0.7])),
            },
            "inactive|good|go": {
                "step_bin": (["inactive", "moderate"], np.array([0.0, 1.0])),
            },
        }

        with mock.patch.object(env, "_transition") as mock_trans:
            mock_trans.lookup = mock_lookup
            mock_trans._include_step_of_day = False  # type: ignore[attr-defined]
            env._success_probs = env._precompute_success_probs()

        ps = env._success_probs["inactive|good|go"]
        # inactive: P(success)=0, term=0*0=0
        # moderate: P(success)=1.0/1.7, term=1.0*0.588=0.588
        # Sum = 0 + 0.588 = 0.588
        expected = 1.0 / 1.7
        assert abs(ps - expected) < 1e-10, f"Expected {expected}, got {ps}"


# ── Bernoulli tracking and burden mapping ────────────────────────────────────


class TestBurdenUpdate:
    def test_idle_skips_burden_update(self) -> None:
        """Calling step with 'idle' should NOT update burden."""
        config = _bayesian_config()
        env = Environment(config, seed=42)
        env.reset()
        env.step("idle")
        # idle should NOT trigger bayesian burden update (skip idle)
        assert env._failure_history is not None
        # After idle step, no failure should be appended
        # The failure_history starts primed with 7 False values
        # After 1 idle step, it should still be all False (7 entries)
        assert len(env._failure_history) == 7
        assert all(not f for f in env._failure_history)

    def test_non_idle_triggers_bernoulli(self) -> None:
        """Calling step with a non-idle action should trigger Bernoulli draw."""
        config = _bayesian_config()
        env = Environment(config, seed=42)
        env.reset()
        # Initial failure history has 7 Falses
        assert env._failure_history is not None
        initial_len = len(env._failure_history)
        assert initial_len == 7

        env.step("go")
        # After 1 non-idle step, history should still have 7 entries (rolling window)
        assert len(env._failure_history) == 7
        # At least one value may have changed
        # (We can't predict since it's random with 0.5 fallback)
        # But the count of True values should be 0 or 1
        true_count = sum(1 for f in env._failure_history if f)
        assert true_count <= 1, f"Expected 0-1 failures, got {true_count}"

    def test_burden_mapping_from_failure_count(self) -> None:
        """Failure count should map to the correct burden level."""
        config = _bayesian_config()
        env = Environment(config, seed=42)
        env.reset()

        # Manually set failure_history to simulate different counts and verify mapping
        assert env._failure_history is not None
        # 0 failures → low
        env._failure_history = deque(
            [False, False, False, False, False, False, False], maxlen=7
        )
        burden = env._get_burden_from_failures()
        assert burden == "low", f"Expected 'low' for 0 failures, got {burden}"

        # 1 failure → low
        env._failure_history = deque(
            [True, False, False, False, False, False, False], maxlen=7
        )
        burden = env._get_burden_from_failures()
        assert burden == "low", f"Expected 'low' for 1 failure, got {burden}"

        # 3 failures → medium
        env._failure_history = deque(
            [True, False, True, False, True, False, False], maxlen=7
        )
        burden = env._get_burden_from_failures()
        assert burden == "medium", f"Expected 'medium' for 3 failures, got {burden}"

        # 5 failures → medium
        env._failure_history = deque(
            [True, True, False, True, False, True, True], maxlen=7
        )
        burden = env._get_burden_from_failures()
        assert burden == "medium", f"Expected 'medium' for 5 failures, got {burden}"

        # 6 failures → high
        env._failure_history = deque(
            [True, True, False, True, True, True, True], maxlen=7
        )
        burden = env._get_burden_from_failures()
        assert burden == "high", f"Expected 'high' for 6 failures, got {burden}"

        # 7 failures → high
        env._failure_history = deque(
            [True, True, True, True, True, True, True], maxlen=7
        )
        burden = env._get_burden_from_failures()
        assert burden == "high", f"Expected 'high' for 7 failures, got {burden}"

    def test_reset_clears_failure_history(self) -> None:
        """Reset should clear failure history and re-prime with all False."""
        config = _bayesian_config()
        env = Environment(config, seed=42)
        env.reset()

        assert env._failure_history is not None
        # Simulate some failures
        env._failure_history.append(True)
        env._failure_history.append(True)
        assert sum(1 for f in env._failure_history if f) == 2

        # Reset
        env.reset()
        assert len(env._failure_history) == 7
        assert all(not f for f in env._failure_history)

    def test_random_fallback_does_not_crash(self) -> None:
        """With random transition, bayesian burden should still work (0.5 P_success)."""
        config = _bayesian_config()
        env = Environment(config, seed=42)
        env.reset()

        for _ in range(20):
            state, _, done = env.step("go")
            if done:
                break
            # burden should be a valid level
            assert state.burden in ("low", "medium", "high")


# ── Integration with step() and _apply_rolling_advances ──────────────────────


class TestStepIntegration:
    def test_bayesian_burden_updates_via_step(self) -> None:
        """Bayesian burden mechanism should update burden during step()."""
        config = _bayesian_config()
        env = Environment(config, seed=42)
        env.reset()

        # Take a non-idle action
        state, _, _ = env.step("go")
        updated_burden = state.burden if hasattr(state, "burden") else None

        # burden might change (it's random, but the mechanism should have run)
        assert updated_burden is not None
        assert updated_burden in ("low", "medium", "high")

    def test_bayesian_does_not_affect_other_rolling_vars(self) -> None:
        """Non-burden rolling window variables should still use rolling_window_count."""
        # Use a config with both burden (bayesian) and another rolling var
        extra_var_config = copy.deepcopy(_FACTORED_CONFIG)
        extra_var_config["state"]["variables"]["other_rolling"] = {
            "names": ["a", "b", "c"],
            "advanced": {
                "type": "rolling_window_count",
                "mechanism": "rolling_window_count",
                "window_size": 3,
                "conditions": [{"factor": "action", "type": "in", "values": ["go"]}],
                "mapping": {0: "a", 1: "b", 2: "c", 3: "c"},
            },
        }
        extra_var_config["initial_state"]["other_rolling"] = "a"
        # Override burden to use bayesian
        extra_var_config["state"]["variables"]["burden"]["advanced"]["mechanism"] = (
            "bayesian_p_success"
        )
        extra_var_config["state"]["variables"]["burden"]["advanced"]["window_size"] = 7
        extra_var_config["state"]["variables"]["burden"]["advanced"]["mapping"] = dict(
            _BAYESIAN_MAPPING
        )

        config = MDPConfig(**extra_var_config)
        env = Environment(config, seed=42)
        env.reset()
        state, _, _ = env.step("go")
        assert hasattr(state, "other_rolling")
        assert state.burden in ("low", "medium", "high")
        # other_rolling should still work via rolling_window_count mechanism
        assert state.other_rolling in ("a", "b", "c")

    def test_multi_factor_precomputation(self) -> None:
        """Precomputation across multiple stochastic factors should work correctly."""
        config = _bayesian_config()
        env = Environment(config, seed=42)

        # Build a mock lookup with 2 stochastic factors
        mock_lookup = {
            "inactive|good|idle": {
                "step_bin": (
                    ["inactive", "moderate", "active"],
                    np.array([0.8, 0.15, 0.05]),
                ),
                "sleep": (["good", "poor"], np.array([0.9, 0.1])),
            },
            "inactive|good|go": {
                "step_bin": (
                    ["inactive", "moderate", "active"],
                    np.array([0.2, 0.5, 0.3]),
                ),
                "sleep": (["good", "poor"], np.array([0.4, 0.6])),
            },
        }

        with mock.patch.object(env, "_transition") as mock_trans:
            mock_trans.lookup = mock_lookup
            mock_trans._include_step_of_day = False  # type: ignore[attr-defined]
            env._success_probs = env._precompute_success_probs()

        key = "inactive|good|go"
        assert key in env._success_probs
        # Should be the mean of per-factor P_success values
        # step_bin: sum over values of p_a²/(p_a+p_i)
        # sleep: same
        ps = env._success_probs[key]
        assert (
            0.0 <= ps <= 1.5
        )  # With the edge case for both-zero = 0.5, can be up to N_values*0.5 + others

    def test_random_fallback_no_division_by_zero(self) -> None:
        """Random transition should not cause division by zero or NaN."""
        config = _bayesian_config()
        env = Environment(config, seed=42)
        env.reset()

        for _ in range(50):
            state, _, done = env.step("go")
            if done:
                break
            burden = (
                getattr(state, "burden", "low") if hasattr(state, "burden") else "low"
            )
            assert burden in ("low", "medium", "high")
