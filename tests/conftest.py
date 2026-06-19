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
        initial_state="sedentary",
        states={
            "sedentary": {"reward": 0.0},
            "active": {"reward": 1.0},
        },
        actions=[
            {"name": "nudge", "burden_penalty": 0.0},
            {"name": "idle", "burden_penalty": 0.0},
        ],
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
        initial_state="sedentary",
        states={
            "sedentary": {"reward": 0.0},
            "active": {"reward": 1.0},
        },
        actions=[
            {"name": "nudge", "burden_penalty": 0.0},
            {"name": "idle", "burden_penalty": 0.0},
        ],
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

    return StateView(activity="sedentary", day=0, step_of_day=0)
