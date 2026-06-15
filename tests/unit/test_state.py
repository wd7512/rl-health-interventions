from rl_health_interventions.config.schemas import ActivityLevel, TimeOfDay
from rl_health_interventions.state import StateView


def test_state_view_carries_activity_time_step():
    sv = StateView(activity=ActivityLevel.ACTIVE, time_of_day=TimeOfDay.MORNING, day=0, step_of_day=0)
    assert sv.activity == ActivityLevel.ACTIVE
    assert sv.time_of_day == TimeOfDay.MORNING
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
        sv = StateView(ActivityLevel.SEDENTARY, TimeOfDay.MORNING, day, step_of_day)
        assert sv.global_step == expected


def test_state_view_is_frozen():
    import pytest
    sv = StateView(ActivityLevel.SEDENTARY, TimeOfDay.MORNING, 0, 0)
    with pytest.raises(AttributeError):
        setattr(sv, "activity", ActivityLevel.ACTIVE)
