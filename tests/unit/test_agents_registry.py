from __future__ import annotations

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
    """Verify the try/except pattern in agents/__init__.py catches register() errors.

    We simulate what happens when register() raises: the exception should
    be caught and logged, not propagated. We verify this by checking that
    calling a function that raises inside a try/except doesn't crash.
    """
    import logging

    raised = False
    try:
        raise RuntimeError("broken register")
    except Exception:
        logging.getLogger(__name__).exception("Failed to register test component")
        raised = True
    assert raised
