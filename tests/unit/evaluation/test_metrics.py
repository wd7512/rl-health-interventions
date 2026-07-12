"""Tests for evaluation.metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from rl_health_interventions.evaluation.metrics import (
    compute_metrics,
    format_comparison_table,
)


class TestComputeMetrics:
    def test_basic_metrics(self):
        """Test basic metric computation."""
        rewards = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],  # seed 1: total=15
                [2.0, 3.0, 4.0, 5.0, 6.0],  # seed 2: total=20
            ]
        )
        metrics = compute_metrics(rewards)

        assert metrics["total_reward"] == 17.5  # mean of [15, 20]
        assert metrics["total_std"] == pytest.approx(2.5, abs=0.1)
        assert metrics["per_step"] == 3.5  # mean of all values
        assert metrics["last50"] == 3.5  # only 5 steps, so same as per_step

    def test_last50_with_longer_trajectory(self):
        """Test last50 metric with 100 steps."""
        rewards = np.ones((10, 100))  # 10 seeds, 100 steps, all 1.0
        rewards[:, -50:] = 2.0  # last 50 steps are 2.0

        metrics = compute_metrics(rewards)
        assert metrics["last50"] == 2.0


class TestFormatComparisonTable:
    def test_format_table(self):
        """Test table formatting."""
        results = {
            "Agent A": {
                "total_reward": 100.0,
                "total_std": 10.0,
                "per_step": 1.0,
                "last50": 1.1,
            },
            "Agent B": {
                "total_reward": 90.0,
                "total_std": 8.0,
                "per_step": 0.9,
                "last50": 0.95,
            },
        }
        table = format_comparison_table(results)
        assert "Agent A" in table
        assert "Agent B" in table
        assert "100.0" in table or "100" in table
