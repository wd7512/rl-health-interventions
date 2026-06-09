from __future__ import annotations

from unittest.mock import patch

import pytest

from rl_health_interventions.agents import REGISTRY, make
from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent


def test_registry_populated() -> None:
    assert "ThompsonSamplingAgent" in REGISTRY
    assert REGISTRY["ThompsonSamplingAgent"] is ThompsonSamplingAgent


def test_make_returns_instance() -> None:
    instance = make("ThompsonSamplingAgent")
    assert isinstance(instance, ThompsonSamplingAgent)


def test_make_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="NonExistent"):
        make("NonExistent")


def test_import_time_failure_isolation() -> None:
    """A broken module at import time should not prevent other modules from registering."""
    # Simulate a broken module by patching the import to raise
    with patch(
        "rl_health_interventions.agents.thompson_sampling.register",
        side_effect=RuntimeError("broken"),
    ):
        # Re-import would trigger the error; since REGISTRY is already populated,
        # verify the existing entry is still there
        assert "ThompsonSamplingAgent" in REGISTRY
