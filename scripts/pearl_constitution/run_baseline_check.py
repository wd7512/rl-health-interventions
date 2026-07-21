#!/usr/bin/env python3
"""Tier 1 — Sanity Checks for PEARL Constitution.

Usage:
    uv run python scripts/pearl_constitution/run_baseline_check.py [--seeds 10] [--persona base]
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from scripts.pearl_constitution.utils import (
    ARM_NAMES,
    compute_arm_daily_steps,
    format_check_result,
    format_matrix,
    load_constitution_config,
    load_reference,
    run_all_arms,
)

logger = logging.getLogger(__name__)

# Reference baseline steps
_PEARL_BASELINE_STEPS = 5618.2

# Tolerance for T1.1 (fractional)
_T1_1_TOLERANCE = 0.15

# Degenerate thresholds for T1.4
_T1_4_MIN_STEPS = 1  # > 0
_T1_4_MAX_STEPS = 30000

# Baseline period: first 7 days
BASELINE_DAYS = 7

# 1-month period: first 30 days
ONE_MONTH_DAYS = 30


def check_t1_1_baseline_stability(
    daily_steps: dict[str, np.ndarray],
    ref_steps: float,
) -> dict:
    """T1.1: Mean steps during 7-day baseline within ±15% of reference."""
    # Use control arm (index 0) for baseline since it receives no interventions
    control_data = daily_steps.get("control", np.array([]))
    if control_data.size == 0:
        return format_check_result(
            "T1.1",
            "Baseline stability",
            False,
            "No control arm data",
            tier=1,
        )
    baseline_window = control_data[:, :BASELINE_DAYS]
    arm_means = np.mean(baseline_window, axis=1)
    grand_mean = float(np.mean(arm_means))
    lower = ref_steps * (1 - _T1_1_TOLERANCE)
    upper = ref_steps * (1 + _T1_1_TOLERANCE)
    passed = lower <= grand_mean <= upper
    detail = (
        f"mean={grand_mean:.1f}, ref={ref_steps:.1f}, range=[{lower:.1f}, {upper:.1f}]"
    )
    return format_check_result("T1.1", "Baseline stability", passed, detail, tier=1)


def check_t1_2_action_differentiation(
    daily_steps: dict[str, np.ndarray],
) -> dict:
    """T1.2: One-way ANOVA across 4 arms on mean daily steps at 1 month, p < 0.01."""
    from scipy import stats as sp_stats

    arm_means: list[np.ndarray] = []
    for arm in ARM_NAMES:
        data = daily_steps.get(arm, np.array([]))
        if data.size == 0:
            return format_check_result(
                "T1.2",
                "Action differentiation",
                False,
                f"Missing arm: {arm}",
                tier=1,
            )
        window = data[:, :ONE_MONTH_DAYS]
        means = np.mean(window, axis=1)
        arm_means.append(means)

    f_stat, p_val = sp_stats.f_oneway(*arm_means)
    passed = p_val < 0.01
    detail = f"F={f_stat:.4f}, p={p_val:.6f}"
    return format_check_result(
        "T1.2",
        "Action differentiation",
        passed,
        detail,
        tier=1,
    )


def check_t1_3_direction_correctness(
    daily_steps: dict[str, np.ndarray],
) -> dict:
    """T1.3: RL > Control in ≥90% of seeds at 1 month."""
    control = daily_steps.get("control", np.array([]))
    rl = daily_steps.get("rl", np.array([]))
    if control.size == 0 or rl.size == 0:
        return format_check_result(
            "T1.3",
            "Direction correctness",
            False,
            "Missing arm data",
            tier=1,
        )
    ctrl_means = np.mean(control[:, :ONE_MONTH_DAYS], axis=1)
    rl_means = np.mean(rl[:, :ONE_MONTH_DAYS], axis=1)
    n_seeds = len(ctrl_means)
    n_correct = int(np.sum(rl_means > ctrl_means))
    pct_correct = n_correct / n_seeds
    passed = pct_correct >= 0.90
    detail = f"RL > Control in {n_correct}/{n_seeds} seeds ({pct_correct:.0%})"
    return format_check_result(
        "T1.3",
        "Direction correctness",
        passed,
        detail,
        tier=1,
    )


def check_t1_4_no_degenerate_trajectories(
    daily_steps: dict[str, np.ndarray],
) -> dict:
    """T1.4: No 0-step or >30,000-step days across any arm or seed."""
    n_degenerate = 0
    n_total = 0
    for arm in ARM_NAMES:
        data = daily_steps.get(arm, np.array([]))
        if data.size == 0:
            continue
        n_total += data.size
        n_degenerate += int(np.sum(data < _T1_4_MIN_STEPS))
        n_degenerate += int(np.sum(data > _T1_4_MAX_STEPS))

    passed = n_degenerate == 0
    detail = (
        f"Degenerate days: {n_degenerate}/{n_total}"
        if n_degenerate > 0
        else f"No degenerate days ({n_total} total)"
    )
    return format_check_result(
        "T1.4",
        "No degenerate trajectories",
        passed,
        detail,
        tier=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PEARL Constitution — Tier 1 Baseline Checks",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--persona", type=str, default="base", help="Persona name")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pearl_constitution.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    logger.info("Loading PEARL reference data...")
    ref = load_reference()
    ref_steps = ref["demographics"]["mean_baseline_steps"]

    logger.info(
        "Loading config (persona=%s, seeds=%d)...",
        args.persona,
        args.seeds,
    )
    config = load_constitution_config(args.persona, args.config)

    logger.info("Running all arms...")
    trajectories = run_all_arms(config, n_seeds=args.seeds)

    logger.info("Computing daily steps...")
    daily_steps = compute_arm_daily_steps(trajectories)

    logger.info("Running Tier 1 checks...")
    results = [
        check_t1_1_baseline_stability(daily_steps, ref_steps),
        check_t1_2_action_differentiation(daily_steps),
        check_t1_3_direction_correctness(daily_steps),
        check_t1_4_no_degenerate_trajectories(daily_steps),
    ]

    matrix = format_matrix(results)
    print(matrix)

    n_pass = sum(1 for r in results if r["passed"])
    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
