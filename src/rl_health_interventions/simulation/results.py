"""Structured results storage for HeartSteps simulation study."""

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


def save_results(results: SimulationResults, path: str) -> None:
    data = {
        "config": results.config,
        "cv_results": [
            {
                "fold": cv.fold,
                "train_indices": [int(i) for i in cv.train_indices],
                "test_indices": [int(i) for i in cv.test_indices],
                "best_gamma": cv.best_gamma,
                "best_w": cv.best_w,
                "participant_results": [
                    {
                        "participant_idx": pr.participant_idx,
                        "proposed_mean": pr.proposed_mean,
                        "ts_bandit_mean": pr.ts_bandit_mean,
                        "improvement": pr.improvement,
                        "proposed_rewards": pr.proposed_rewards,
                        "ts_bandit_rewards": pr.ts_bandit_rewards,
                    }
                    for pr in cv.participant_results
                ],
            }
            for cv in results.cv_results
        ],
        "summary": results.compute_summary(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", path)
