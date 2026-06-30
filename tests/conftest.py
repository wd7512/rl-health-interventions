from __future__ import annotations

import pathlib

import pytest

from rl_health_interventions.config.schemas import MDPConfig


_HERE = pathlib.Path(__file__).parent
TABLES_DIR = str(_HERE / "assets" / "tables")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    unit_dir = (_HERE / "unit").resolve()
    integration_dir = (_HERE / "integration").resolve()
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
        initial_state={"activity_level": "sedentary"},
        state={
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        actions=["nudge", "idle"],
        reward={
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        transition_model={"type": "rule_based", "table_dir": TABLES_DIR},
        agents=[{"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1}],
    )


@pytest.fixture
def minimal_config() -> MDPConfig:
    return MDPConfig(
        episode_days=1,
        steps_per_day=1,
        seed=42,
        initial_state={"activity_level": "sedentary"},
        state={
            "factors": {
                "activity_level": {"dims": 2, "names": ["sedentary", "active"]},
            },
        },
        actions=["nudge", "idle"],
        reward={
            "factor": "activity_level",
            "values": {"sedentary": 0.0, "active": 1.0},
        },
        transition_model={"type": "rule_based", "table_dir": TABLES_DIR},
    )


@pytest.fixture
def state_view():
    from rl_health_interventions.state import StateView

    return StateView({"activity_level": "sedentary"}, day=0, step_of_day=0)


@pytest.fixture
def sprint1_config() -> MDPConfig:
    return MDPConfig(
        episode_days=1,
        steps_per_day=5,
        seed=42,
        initial_state={
            "step_bin": "inactive",
            "sleep": "rested",
            "day_of_week": "weekday",
            "burden": "low",
        },
        state={
            "factors": {
                "step_bin": {"dims": 3, "names": ["inactive", "moderate", "active"]},
                "sleep": {"dims": 2, "names": ["rested", "under_rested"]},
                "day_of_week": {"dims": 2, "names": ["weekday", "weekend"]},
                "burden": {"dims": 3, "names": ["low", "medium", "high"]},
            },
        },
        actions={
            "idle": {"action_penalty": 0},
            "movement_suggestion": {"action_penalty": 1},
            "goal_reminder": {"action_penalty": 1},
            "journal": {"action_penalty": 1},
        },
        reward={
            "factor": "step_bin",
            "values": {"inactive": 0.0, "moderate": 0.5, "active": 1.0},
            "action_penalty_multiplier": 0.05,
        },
        transition_model={"type": "rule_based", "table_dir": TABLES_DIR},
    )
