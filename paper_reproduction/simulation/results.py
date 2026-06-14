"""Structured results storage for simulation study.

Provides the SimulationResults dataclass for collecting cross-validation
output, compute_summary() for aggregate statistics, and JSON serialization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParticipantResult:
    """Results for one participant in one CV fold."""

    participant_idx: int
    proposed_mean: float
    ts_bandit_mean: float
    improvement: float
    proposed_rewards: list[float] = field(default_factory=list)
    ts_bandit_rewards: list[float] = field(default_factory=list)


@dataclass
class CVFoldResult:
    """Results for one cross-validation fold."""

    fold: int
    train_indices: list[int]
    test_indices: list[int]
    best_gamma: float
    best_w: float
    participant_results: list[ParticipantResult]


@dataclass
class SimulationResults:
    """Complete simulation study results."""

    config: dict[str, Any]
    cv_results: list[CVFoldResult]

    def compute_summary(self) -> dict[str, Any]:
        """Aggregate statistics across all participants.

        Returns:
            Dict with n_participants, n_improved, pct_improved,
            mean/median/std/min/max improvement.
        """
        all_improvements = []
        for cv in self.cv_results:
            for pr in cv.participant_results:
                all_improvements.append(pr.improvement)

        arr = np.array(all_improvements)
        n_total = len(arr)
        n_improved = int(np.sum(arr > 0))

        return {
            "n_participants": n_total,
            "n_improved": n_improved,
            "pct_improved": float(n_improved / max(n_total, 1) * 100),
            "mean_improvement": float(np.mean(arr)) if n_total > 0 else 0.0,
            "median_improvement": float(np.median(arr)) if n_total > 0 else 0.0,
            "std_improvement": float(np.std(arr, ddof=1)) if n_total > 1 else 0.0,
            "min_improvement": float(np.min(arr)) if n_total > 0 else 0.0,
            "max_improvement": float(np.max(arr)) if n_total > 0 else 0.0,
        }


def _participant_to_dict(pr: ParticipantResult) -> dict[str, Any]:
    return {
        "participant_idx": pr.participant_idx,
        "proposed_mean": pr.proposed_mean,
        "ts_bandit_mean": pr.ts_bandit_mean,
        "improvement": pr.improvement,
        "proposed_rewards": pr.proposed_rewards,
        "ts_bandit_rewards": pr.ts_bandit_rewards,
    }


def _cvfold_to_dict(cv: CVFoldResult) -> dict[str, Any]:
    return {
        "fold": cv.fold,
        "train_indices": [int(i) for i in cv.train_indices],
        "test_indices": [int(i) for i in cv.test_indices],
        "best_gamma": cv.best_gamma,
        "best_w": cv.best_w,
        "participant_results": [
            _participant_to_dict(pr) for pr in cv.participant_results
        ],
    }


def save_results(results: SimulationResults, path: str) -> None:
    """Serialize SimulationResults to JSON.

    Args:
        results: SimulationResults instance.
        path: Output file path.
    """
    data = {
        "config": results.config,
        "cv_results": [_cvfold_to_dict(cv) for cv in results.cv_results],
        "summary": results.compute_summary(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", path)


def load_results(path: str) -> dict[str, Any]:
    """Load simulation results from a JSON file.

    Args:
        path: Path to JSON results file.

    Returns:
        Dict with keys 'config', 'cv_results', 'summary'.
    """
    with open(path) as f:
        return dict(json.load(f))
