"""Metrics computation and formatting for evaluation benchmarks."""

from __future__ import annotations

import numpy as np


def compute_metrics(rewards: np.ndarray) -> dict[str, float]:
    """Compute standard metrics from per-step rewards.

    Parameters
    ----------
    rewards : ndarray of shape (n_seeds, n_steps)
        Per-step reward for each seed and time step.

    Returns
    -------
    dict[str, float]
        Keys: ``total_reward``, ``total_std``, ``per_step``, ``last50``.
    """
    return {
        "total_reward": float(rewards.sum(axis=1).mean()),
        "total_std": float(rewards.sum(axis=1).std()),
        "per_step": float(rewards.mean()),
        "last50": float(rewards[:, -50:].mean()),
    }


def format_comparison_table(results: dict[str, dict]) -> str:
    """Format agent results as a comparison table string suitable for logging.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping from agent label to dict with keys ``total_reward``,
        ``total_std``, ``per_step``, ``last50``.

    Returns
    -------
    str
        Multi-line table string.
    """
    header = f"{'Agent':<20} {'Total Reward':>14} {'Per Step':>10} {'Last 50':>10}"
    sep = "-" * 72
    lines = [sep, header, sep]

    sorted_labels = sorted(results, key=lambda k: results[k]["per_step"], reverse=True)
    for label in sorted_labels:
        r = results[label]
        lines.append(
            f"{label:<20} {r['total_reward']:>8.1f} +- {r['total_std']:<5.1f}"
            f" {r['per_step']:>9.4f} {r['last50']:>10.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)
