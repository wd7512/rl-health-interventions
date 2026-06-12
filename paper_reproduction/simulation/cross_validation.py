"""3-fold cross-validation over participants.

Implements the cross-validation procedure described in Section 6 of
the HeartSteps V2 paper.

Partitions participants into approximately equal folds. For each of the
3 iterations, one fold is held out for testing and the remaining folds
form the training batch.

Reference:
    Liao et al. (2019). arXiv:1909.03539, Section 6.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def create_folds(
    n_participants: int,
    n_folds: int = 3,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create n_folds cross-validation splits over participants.

    Each participant appears in exactly one test fold across all iterations.
    Training and test folds are disjoint in every iteration.

    Args:
        n_participants: Total number of participants to split.
        n_folds: Number of folds (default 3 per the paper).
        rng: NumPy random generator. Created if None.

    Returns:
        List of (train_indices, test_indices) tuples, one per fold.
        Each index array is 1-D with dtype int.
    """
    if rng is None:
        rng = np.random.default_rng()

    indices = np.arange(n_participants, dtype=int)
    rng.shuffle(indices)

    folds = np.array_split(indices, n_folds)

    result: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        test_indices = folds[i]
        train_indices = np.concatenate(
            [folds[j] for j in range(n_folds) if j != i]
        )
        result.append((train_indices, test_indices))

    logger.debug(
        "Created %d folds: fold sizes=%s",
        n_folds,
        [len(f) for f in folds],
    )

    return result
