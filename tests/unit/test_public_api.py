from __future__ import annotations

import pytest

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


@pytest.mark.parametrize(
    "func",
    [make_transition, make_reward, make_agent, make_response_model, make_dataset],
)
def test_make_functions_imported(func):
    assert callable(func)
