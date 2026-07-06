from __future__ import annotations

import pathlib

import pytest

from rl_health_interventions.config.schemas import MDPConfig


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
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
        episode_days=90,
        steps_per_day=5,
        seed=42,
        state={"variables": {"activity_level": {"names": ["sedentary", "active"]}}},
        initial_state={"activity_level": "sedentary"},
        actions=["nudge", "idle"],
        reward={
            "variables": {
                "value": {
                    "source": "state.activity_level",
                    "mapping": {"sedentary": 0.0, "active": 1.0},
                }
            },
            "formula": "value",
        },
        transition_model={
            "type": "rule_based",
            "transition_probabilities": {
                "sedentary": {
                    "nudge": {"active": 0.3, "sedentary": 0.7},
                    "idle": {"active": 0.1, "sedentary": 0.9},
                },
                "active": {
                    "nudge": {"active": 0.5, "sedentary": 0.5},
                    "idle": {"active": 0.6, "sedentary": 0.4},
                },
            },
        },
        agents=[{"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1}],
    )


@pytest.fixture
def minimal_config() -> MDPConfig:
    return MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        state={"variables": {"activity_level": {"names": ["sedentary", "active"]}}},
        initial_state={"activity_level": "sedentary"},
        actions=["nudge", "idle"],
        reward={
            "variables": {
                "value": {
                    "source": "state.activity_level",
                    "mapping": {"sedentary": 0.0, "active": 1.0},
                }
            },
            "formula": "value",
        },
        transition_model={
            "type": "rule_based",
            "transition_probabilities": {
                "sedentary": {
                    "nudge": {"active": 0.5, "sedentary": 0.5},
                    "idle": {"active": 0.5, "sedentary": 0.5},
                },
                "active": {
                    "nudge": {"active": 0.5, "sedentary": 0.5},
                    "idle": {"active": 0.5, "sedentary": 0.5},
                },
            },
        },
        agents=[],
    )


@pytest.fixture
def state_view():
    from rl_health_interventions.state import StateView

    return StateView(factors={"activity_level": "sedentary"}, day=0, step_of_day=0)


@pytest.fixture
def sedentary_state():
    from rl_health_interventions.state import StateView

    return StateView(factors={"activity_level": "sedentary"}, day=0, step_of_day=0)


@pytest.fixture
def active_state():
    from rl_health_interventions.state import StateView

    return StateView(factors={"activity_level": "active"}, day=0, step_of_day=0)


@pytest.fixture
def sed_and_act():
    from rl_health_interventions.state import StateView

    return (
        StateView(factors={"activity_level": "sedentary"}, day=0, step_of_day=0),
        StateView(factors={"activity_level": "active"}, day=0, step_of_day=0),
    )


@pytest.fixture
def sprint1_config() -> MDPConfig:
    return MDPConfig(
        episode_days=90,
        steps_per_day=5,
        seed=42,
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
            "formula": "alpha * step_bin_value + "
            "(1 - alpha) * sleep_value - action_penalty",
        },
        transition_model={"type": "random"},
        agents=[],
    )
