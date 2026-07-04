from rl_health_interventions.state import StateView


def test_state_view_carries_factors():
    sv = StateView(factors={"activity_level": "active"}, day=0, step_of_day=0)
    assert sv.activity_level == "active"
    assert sv.day == 0
    assert sv.step_of_day == 0


def test_state_view_global_step_calculation():
    sv = StateView(
        factors={"activity_level": "sedentary"}, day=1, step_of_day=2, steps_per_day=5
    )
    assert sv.global_step == 7


def test_state_view_is_immutable():
    import pytest

    sv = StateView(factors={"activity_level": "sedentary"}, day=0, step_of_day=0)
    with pytest.raises(AttributeError, match="immutable"):
        setattr(sv, "activity_level", "active")


def test_state_view_unknown_factor_raises():
    import pytest

    sv = StateView(factors={"activity_level": "sedentary"}, day=0, step_of_day=0)
    with pytest.raises(AttributeError, match="nonexistent"):
        sv.nonexistent  # noqa: B018


def test_state_view_with_factors_returns_new_instance():
    sv = StateView(factors={"activity_level": "sedentary"}, day=0, step_of_day=0)
    sv2 = sv.with_factors(activity_level="active")
    assert sv.activity_level == "sedentary"
    assert sv2.activity_level == "active"
    assert sv2.day == 0


def test_state_view_state_key():
    sv = StateView(
        factors={"activity_level": "active", "sleep": "good"}, day=0, step_of_day=0
    )
    assert sv.state_key == ("active", "good")


def test_state_view_equality():
    sv1 = StateView(factors={"activity_level": "active"}, day=0, step_of_day=0)
    sv2 = StateView(factors={"activity_level": "active"}, day=0, step_of_day=0)
    sv3 = StateView(factors={"activity_level": "sedentary"}, day=0, step_of_day=0)
    assert sv1 == sv2
    assert sv1 != sv3


def test_state_view_hashable():
    sv1 = StateView(factors={"activity_level": "active"}, day=0, step_of_day=0)
    sv2 = StateView(factors={"activity_level": "active"}, day=0, step_of_day=0)
    assert hash(sv1) == hash(sv2)


def test_state_view_factor_values():
    sv = StateView(
        factors={"activity_level": "active", "sleep": "good"}, day=0, step_of_day=0
    )
    assert sv.factor_values == {"activity_level": "active", "sleep": "good"}
