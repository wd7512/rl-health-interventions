from __future__ import annotations

from rl_health_interventions import (
    __version__,
    make_agent,
    make_dataset,
    make_response_model,
    make_reward,
    make_transition,
)


def test_version_exists() -> None:
    # Version is mirrored from pyproject.toml; must be a non-empty semver string.
    assert __version__
    parts = __version__.split(".")
    assert len(parts) == 3, f"expected semver, got {__version__!r}"


def test_make_transition_imported() -> None:
    assert callable(make_transition)


def test_make_reward_imported() -> None:
    assert callable(make_reward)


def test_make_agent_imported() -> None:
    assert callable(make_agent)


def test_make_response_model_imported() -> None:
    assert callable(make_response_model)


def test_make_dataset_imported() -> None:
    assert callable(make_dataset)
