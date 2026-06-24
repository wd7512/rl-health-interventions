"""Tests for HeartSteps config validation."""

from __future__ import annotations

import pytest

from rl_health_interventions.config.schemas import AgentConfig


class TestHeartStepsAgentConfig:
    def test_valid_heartsteps_config(self) -> None:
        cfg = AgentConfig(type="heartsteps", tau=1.0, epsilon_0=0.2, epsilon_1=0.1)
        assert cfg.type == "heartsteps"

    def test_heartsteps_rejects_alpha_prior(self) -> None:
        with pytest.raises(ValueError, match="does not accept alpha_prior"):
            AgentConfig(type="heartsteps", alpha_prior=1.0)

    def test_heartsteps_rejects_epsilon(self) -> None:
        with pytest.raises(ValueError, match="does not accept epsilon"):
            AgentConfig(type="heartsteps", epsilon=0.1)
