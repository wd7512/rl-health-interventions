from __future__ import annotations

import pytest

from rl_health_interventions.state import StateView


class TestStateViewConstruction:
    def test_single_factor_mvp(self):
        sv = StateView(
            {"activity_level": "sedentary"}, day=0, step_of_day=0, steps_per_day=5
        )
        assert sv.activity_level == "sedentary"
        assert sv.day == 0
        assert sv.step_of_day == 0
        assert sv.steps_per_day == 5
        assert sv.factor_names == ("activity_level",)

    def test_multi_factor_sprint1(self):
        factors = {
            "step_bin": "inactive",
            "sleep": "rested",
            "day_of_week": "weekday",
            "burden": "low",
        }
        sv = StateView(factors, day=0, step_of_day=0, steps_per_day=5)
        assert sv.step_bin == "inactive"
        assert sv.sleep == "rested"
        assert sv.day_of_week == "weekday"
        assert sv.burden == "low"

    def test_default_steps_per_day(self):
        sv = StateView({"activity_level": "sedentary"})
        assert sv.steps_per_day == 5


class TestGetAttrAccess:
    def test_getattr_returns_factor_value(self):
        sv = StateView({"activity_level": "active"})
        assert sv.activity_level == "active"

    def test_getattr_raises_for_unknown_factor(self):
        sv = StateView({"activity_level": "sedentary"})
        with pytest.raises(AttributeError, match="unknown_factor"):
            _ = sv.unknown_factor

    def test_getattr_error_lists_available_factors(self):
        sv = StateView({"activity_level": "sedentary"})
        with pytest.raises(AttributeError, match="activity_level"):
            _ = sv.nonexistent


class TestProperties:
    def test_global_step(self):
        sv = StateView(
            {"activity_level": "sedentary"}, day=1, step_of_day=3, steps_per_day=5
        )
        assert sv.global_step == 8  # 1*5 + 3

    def test_global_step_zero(self):
        sv = StateView(
            {"activity_level": "sedentary"}, day=0, step_of_day=0, steps_per_day=5
        )
        assert sv.global_step == 0

    def test_state_key_single_factor(self):
        sv = StateView({"activity_level": "active"})
        assert sv.state_key == ("active",)

    def test_state_key_multi_factor(self):
        factors = {
            "step_bin": "moderate",
            "sleep": "under_rested",
            "day_of_week": "weekend",
            "burden": "high",
        }
        sv = StateView(factors)
        # state_key is sorted by factor name alphabetically
        assert sv.state_key == ("high", "weekend", "under_rested", "moderate")

    def test_state_key_order_matches_factor_names(self):
        sv = StateView({"a": "x", "b": "y"})
        assert sv.state_key == ("x", "y")

    def test_factor_names(self):
        sv = StateView({"step_bin": "active", "sleep": "rested"})
        assert sv.factor_names == ("step_bin", "sleep")


class TestWithFactors:
    def test_with_factors_returns_new_instance(self):
        sv = StateView(
            {"activity_level": "sedentary"}, day=1, step_of_day=2, steps_per_day=5
        )
        sv2 = sv.with_factors(activity_level="active")
        assert sv2.activity_level == "active"
        assert sv.activity_level == "sedentary"
        assert sv2.day == sv.day
        assert sv2.step_of_day == sv.step_of_day

    def test_with_factors_partial_update(self):
        factors = {"step_bin": "inactive", "sleep": "rested"}
        sv = StateView(factors)
        sv2 = sv.with_factors(sleep="under_rested")
        assert sv2.step_bin == "inactive"
        assert sv2.sleep == "under_rested"

    def test_with_factors_preserves_day_step(self):
        sv = StateView({"a": "x"}, day=3, step_of_day=4, steps_per_day=5)
        sv2 = sv.with_factors(a="y")
        assert sv2.day == 3
        assert sv2.step_of_day == 4
        assert sv2.steps_per_day == 5


class TestWithAdvance:
    def test_with_advance_updates_day(self):
        sv = StateView({"activity_level": "sedentary"}, day=0, step_of_day=0)
        sv2 = sv.with_advance(day=1)
        assert sv2.day == 1
        assert sv2.step_of_day == 0

    def test_with_advance_updates_step_of_day(self):
        sv = StateView({"activity_level": "sedentary"}, day=0, step_of_day=0)
        sv2 = sv.with_advance(step_of_day=3)
        assert sv2.day == 0
        assert sv2.step_of_day == 3

    def test_with_advance_preserves_factors(self):
        sv = StateView({"activity_level": "active"}, day=0, step_of_day=0)
        sv2 = sv.with_advance(day=1)
        assert sv2.activity_level == "active"

    def test_with_advance_none_params_keeps_originals(self):
        sv = StateView({"a": "x"}, day=1, step_of_day=2)
        sv2 = sv.with_advance()
        assert sv2.day == 1
        assert sv2.step_of_day == 2

    def test_with_advance_preserves_steps_per_day(self):
        sv = StateView({"a": "x"}, day=0, step_of_day=0, steps_per_day=10)
        sv2 = sv.with_advance(day=1)
        assert sv2.steps_per_day == 10


class TestEqualityAndHashing:
    def test_equality_same_factors_day_step(self):
        sv1 = StateView({"a": "x"}, day=0, step_of_day=0)
        sv2 = StateView({"a": "x"}, day=0, step_of_day=0)
        assert sv1 == sv2

    def test_inequality_different_factors(self):
        sv1 = StateView({"a": "x"}, day=0, step_of_day=0)
        sv2 = StateView({"a": "y"}, day=0, step_of_day=0)
        assert sv1 != sv2

    def test_inequality_different_day(self):
        sv1 = StateView({"a": "x"}, day=0, step_of_day=0)
        sv2 = StateView({"a": "x"}, day=1, step_of_day=0)
        assert sv1 != sv2

    def test_inequality_different_step_of_day(self):
        sv1 = StateView({"a": "x"}, day=0, step_of_day=0)
        sv2 = StateView({"a": "x"}, day=0, step_of_day=1)
        assert sv1 != sv2

    def test_hash_equal_for_equal_states(self):
        sv1 = StateView({"a": "x"}, day=0, step_of_day=0)
        sv2 = StateView({"a": "x"}, day=0, step_of_day=0)
        assert hash(sv1) == hash(sv2)

    def test_hash_differs_for_different_states(self):
        sv1 = StateView({"a": "x"}, day=0, step_of_day=0)
        sv2 = StateView({"a": "y"}, day=0, step_of_day=0)
        assert hash(sv1) != hash(sv2)

    def test_equality_with_non_stateview(self):
        sv = StateView({"a": "x"})
        assert sv != "not a state"
        assert sv != 42
        assert sv is not None


class TestImmutability:
    def test_cannot_modify_factor_via_setattr(self):
        sv = StateView({"activity_level": "sedentary"})
        with pytest.raises(AttributeError):
            sv.activity_level = "active"

    def test_cannot_modify_day(self):
        sv = StateView({"a": "x"}, day=0, step_of_day=0)
        with pytest.raises(AttributeError):
            sv.day = 1

    def test_cannot_modify_step_of_day(self):
        sv = StateView({"a": "x"}, day=0, step_of_day=0)
        with pytest.raises(AttributeError):
            sv.step_of_day = 1


class TestRepr:
    def test_repr_single_factor(self):
        sv = StateView({"activity_level": "sedentary"}, day=0, step_of_day=0)
        r = repr(sv)
        assert "activity_level=sedentary" in r
        assert "day=0" in r
        assert "step_of_day=0" in r

    def test_repr_multi_factor(self):
        sv = StateView({"a": "x", "b": "y"}, day=1, step_of_day=2)
        r = repr(sv)
        assert "a=x" in r
        assert "b=y" in r
        assert "day=1" in r
        assert "step_of_day=2" in r
