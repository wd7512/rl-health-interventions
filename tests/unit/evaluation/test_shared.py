"""Tests for evaluation._shared module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from rl_health_interventions.evaluation._shared import (
    agent_label,
    resolve_config,
    run_agent,
)


class TestResolveConfig:
    def test_absolute_path_passthrough(self):
        """Absolute paths should pass through unchanged."""
        result = resolve_config(Path("/tmp"), "/absolute/path.yaml")
        assert result == "/absolute/path.yaml"

    def test_bare_name_uses_config_dir(self):
        """Bare names should be resolved relative to config_dir."""
        result = resolve_config(Path("/configs"), "test.yaml")
        assert result == "/configs/test.yaml"

    def test_none_uses_default(self):
        """None should use the default config name."""
        result = resolve_config(Path("/configs"), default="default.yaml")
        assert result == "/configs/default.yaml"


class TestAgentLabel:
    def test_random_agent(self):
        """Random agent should return 'Random'."""
        cfg = MagicMock()
        cfg.type = "random"
        assert agent_label(cfg) == "Random"

    def test_contextual_thompson(self):
        """Contextual TS should return 'Ctx TS'."""
        cfg = MagicMock()
        cfg.type = "thompson_sampling"
        cfg.contextual = True
        assert agent_label(cfg) == "Ctx TS"

    def test_standard_epsilon_greedy(self):
        """Standard EG should return 'Std EG'."""
        cfg = MagicMock()
        cfg.type = "epsilon_greedy"
        cfg.contextual = False
        assert agent_label(cfg) == "Std EG"

    def test_standard_ppo(self):
        cfg = MagicMock()
        cfg.type = "ppo"
        cfg.contextual = False
        assert agent_label(cfg) == "Std PPO"


class TestRunAgent:
    def test_returns_correct_shape(self):
        """run_agent should return (n_seeds, n_steps) array."""
        config = MagicMock()
        config.action_names = ["idle", "nudge"]

        agent_cfg = MagicMock()
        agent_cfg.type = "random"
        agent_cfg.contextual = False
        agent_cfg.model_dump.return_value = {"type": "random"}

        with (
            patch("rl_health_interventions.evaluation._shared.run_episode") as mock_run,
            patch("rl_health_interventions.evaluation._shared.make_agent") as mock_make,
            patch(
                "rl_health_interventions.evaluation._shared.derive_agent_seed"
            ) as mock_seed,
        ):
            mock_run.return_value = [{"reward": 1.0}] * 10  # 10 steps
            mock_make.return_value = MagicMock()
            mock_seed.return_value = 42
            result = run_agent(config, agent_cfg, n_seeds=3)

        assert result.shape == (3, 10)
        assert isinstance(result, np.ndarray)
