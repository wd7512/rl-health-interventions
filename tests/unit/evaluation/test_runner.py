"""Tests for evaluation.runner module."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from rl_health_interventions.evaluation.runner import _positive_int, run_benchmark


class TestPositiveInt:
    def test_valid_positive(self):
        assert _positive_int("50") == 50

    def test_zero_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int("0")

    def test_negative_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int("-5")


class TestRunBenchmark:
    def test_returns_metrics_dict(self):
        """run_benchmark should return {agent_label: metrics_dict}."""
        # Mock load_config and run_agent
        with patch(
            "rl_health_interventions.evaluation.runner.load_config"
        ) as mock_load:
            mock_config = MagicMock()
            mock_config.seed = 42
            mock_config.episode_days = 10
            mock_config.steps_per_day = 5
            mock_config.agents = []
            mock_load.return_value = mock_config

            result = run_benchmark("/fake/config.yaml", n_seeds=5)
            assert isinstance(result, dict)

    def test_writes_json_fixture(self, tmp_path):
        """run_benchmark should write JSON when dump_json=True."""
        with patch(
            "rl_health_interventions.evaluation.runner.load_config"
        ) as mock_load:
            mock_config = MagicMock()
            mock_config.seed = 42
            mock_config.episode_days = 10
            mock_config.steps_per_day = 5
            mock_config.agents = []
            mock_load.return_value = mock_config

            run_benchmark(
                "/fake/config.yaml",
                n_seeds=5,
                output_dir=tmp_path,
                dump_json=True,
                confirm_overwrite=True,
            )

            # Check that a JSON file was written
            json_files = list(tmp_path.glob("*.json"))
            assert len(json_files) == 1
