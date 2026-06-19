from rl_health_interventions.state import StateView


def test_state_view_carries_activity_and_step():
    sv = StateView(activity="active", day=0, step_of_day=0)
    assert sv.activity == "active"
    assert sv.day == 0
    assert sv.step_of_day == 0


def test_state_view_global_step_calculation():
    cases = [
        (0, 0, 0),
        (0, 4, 4),
        (1, 0, 5),
        (2, 3, 13),
        (89, 4, 449),
    ]
    for day, step_of_day, expected in cases:
        sv = StateView("sedentary", day, step_of_day)
        assert sv.global_step == expected


def test_state_view_is_frozen():
    import pytest

    sv = StateView("sedentary", 0, 0)
    with pytest.raises(AttributeError):
        setattr(sv, "activity", "active")


def test_state_view_full_construction():
    sv = StateView(
        activity="active",
        day=2,
        step_of_day=3,
        steps_per_day=10,
        steps=5000.0,
        weight=71.5,
        time_of_day=7,
        day_of_week=2,
    )
    assert sv.activity == "active"
    assert sv.day == 2
    assert sv.step_of_day == 3
    assert sv.steps_per_day == 10
    assert sv.steps == 5000.0
    assert sv.weight == 71.5
    assert sv.time_of_day == 7
    assert sv.day_of_week == 2
    assert sv.global_step == 23
    import pytest

    with pytest.raises(AttributeError):
        setattr(sv, "steps", 6000.0)


def test_state_view_optional_fields_default_to_none():
    sv = StateView(activity="sedentary", day=0, step_of_day=0)
    assert sv.steps is None
    assert sv.weight is None
    assert sv.time_of_day is None
    assert sv.day_of_week is None
