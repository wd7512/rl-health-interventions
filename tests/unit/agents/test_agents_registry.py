from __future__ import annotations

from rl_health_interventions.agents import REGISTRY, make
from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent


def test_registry_populated() -> None:
    assert "ThompsonSamplingAgent" in REGISTRY
    assert REGISTRY["ThompsonSamplingAgent"] is ThompsonSamplingAgent


def test_make_returns_instance() -> None:
    instance = make("ThompsonSamplingAgent")
    assert isinstance(instance, ThompsonSamplingAgent)


def test_make_unknown_raises_keyerror() -> None:
    try:
        make("NonExistent")
        assert False, "Expected KeyError"
    except KeyError as e:
        assert "NonExistent" in str(e)
        assert "ThompsonSamplingAgent" in str(e)


def test_import_time_failure_isolation() -> None:
    assert "ThompsonSamplingAgent" in REGISTRY
