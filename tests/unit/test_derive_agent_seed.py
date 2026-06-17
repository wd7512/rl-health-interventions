"""Unit tests for agent seed derivation."""

from rl_health_interventions.agents import derive_agent_seed


def test_deterministic_same_inputs():
    assert derive_agent_seed(42) == derive_agent_seed(42)


def test_different_base_seeds_differ():
    assert derive_agent_seed(42) != derive_agent_seed(43)


def test_different_indices_differ():
    assert derive_agent_seed(42, 0) != derive_agent_seed(42, 1)


def test_index_zero_is_not_base_seed():
    """Agent seed must differ from environment seed to avoid correlated RNG."""
    assert derive_agent_seed(42, 0) != 42


def test_output_in_valid_range():
    result = derive_agent_seed(42)
    assert 0 <= result < 2**31
