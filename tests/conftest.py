from __future__ import annotations

import pathlib

import pytest

from rl_health_interventions.config.schemas import (
    ActivityLevel,
    Action,
    MDPConfig,
    TimeOfDay,
    TimeOfDayMask,
    TransitionMatrix,
)

ALL_TIMES = [
    TimeOfDay.MORNING,
    TimeOfDay.MIDDAY,
    TimeOfDay.AFTERNOON,
    TimeOfDay.EVENING,
    TimeOfDay.NIGHT,
]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-apply markers based on test file location.

    Respects explicit markers — if a test already has a marker, don't override.
    """
    unit_dir = (pathlib.Path(__file__).parent / "unit").resolve()
    integration_dir = (pathlib.Path(__file__).parent / "integration").resolve()
    for item in items:
        test_path = pathlib.Path(item.path).resolve()
        existing_markers = {m.name for m in item.iter_markers()}
        if test_path.is_relative_to(unit_dir) and "integration" not in existing_markers:
            item.add_marker(pytest.mark.unit)
        elif (
            test_path.is_relative_to(integration_dir) and "unit" not in existing_markers
        ):
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def valid_config() -> MDPConfig:
    return MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=ALL_TIMES,
        steps_per_day=5,
        episode_days=90,
        transition=TransitionMatrix(
            root={
                ActivityLevel.SEDENTARY: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.7,
                        ActivityLevel.ACTIVE: 0.3,
                    },
                    Action.DON_T_SEND: {
                        ActivityLevel.SEDENTARY: 0.9,
                        ActivityLevel.ACTIVE: 0.1,
                    },
                },
                ActivityLevel.ACTIVE: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.2,
                        ActivityLevel.ACTIVE: 0.8,
                    },
                    Action.DON_T_SEND: {
                        ActivityLevel.SEDENTARY: 0.4,
                        ActivityLevel.ACTIVE: 0.6,
                    },
                },
            }
        ),
        masks=TimeOfDayMask(
            root={
                t: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0}
                for t in ALL_TIMES[:-1]
            }
            | {
                TimeOfDay.NIGHT: {
                    ActivityLevel.SEDENTARY: 1.0,
                    ActivityLevel.ACTIVE: 1.0,
                }
            }
        ),
        seed=42,
    )


@pytest.fixture
def minimal_config() -> MDPConfig:
    return MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=[TimeOfDay.MORNING],
        steps_per_day=1,
        episode_days=1,
        transition=TransitionMatrix(
            root={
                ActivityLevel.SEDENTARY: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.5,
                        ActivityLevel.ACTIVE: 0.5,
                    },
                    Action.DON_T_SEND: {
                        ActivityLevel.SEDENTARY: 0.5,
                        ActivityLevel.ACTIVE: 0.5,
                    },
                },
                ActivityLevel.ACTIVE: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.5,
                        ActivityLevel.ACTIVE: 0.5,
                    },
                    Action.DON_T_SEND: {
                        ActivityLevel.SEDENTARY: 0.5,
                        ActivityLevel.ACTIVE: 0.5,
                    },
                },
            }
        ),
        masks=TimeOfDayMask(
            root={
                TimeOfDay.MORNING: {
                    ActivityLevel.SEDENTARY: 0.0,
                    ActivityLevel.ACTIVE: 0.0,
                }
            }
        ),
    )
