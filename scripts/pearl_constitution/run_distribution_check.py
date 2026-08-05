#!/usr/bin/env python3
"""Tier 2 — Distribution Checks for PEARL Constitution.

Usage:
    uv run python scripts/pearl_constitution/run_distribution_check.py [--seeds 10] [--persona base]
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
from scipy import stats as sp_stats

from scripts.pearl_constitution.utils import (
    ARM_NAMES,
    compute_arm_daily_steps,
    format_check_result,
    format_matrix,
    icc_1way_random,
    load_constitution_config,
    load_reference,
    run_all_arms,
)

logger = logging.getLogger(__name__)

BASELINE_DAYS = 7
ONE_MONTH_DAYS = 30
TWO_MONTH_DAYS = 60


def check_t2_1_baseline_mean(
    daily_steps: dict[str, np.ndarray], ref_steps: float
) -> dict:
    """T2.1: One-sample t-test vs μ=5,618 on 7-day baseline, p > 0.05."""
    control = daily_steps.get("control", np.array([]))
    if control.size == 0:
        return format_check_result(
            "T2.1",
            "Baseline mean",
            False,
            "No control arm data",
            tier=2,
        )
    baseline_means = np.mean(control[:, :BASELINE_DAYS], axis=1)
    t_stat, p_val = sp_stats.ttest_1samp(baseline_means, ref_steps)
    passed = p_val > 0.05
    detail = (
        f"t={t_stat:.4f}, p={p_val:.4f}, "
        f"mean={np.mean(baseline_means):.1f}, ref={ref_steps:.1f}"
    )
    return format_check_result("T2.1", "Baseline mean", passed, detail, tier=2)


def check_t2_2_effect_size_magnitude(
    daily_steps: dict[str, np.ndarray],
    ref: dict,
) -> dict:
    """T2.2: RL vs Control Δ at 1 month within the reference-derived band.

    The band is derived from the reference 1-month effect-size deltas
    (observed range 218-296: RL vs Fixed Δ=238, RL vs Random Δ=218, RL vs
    Control Δ=296). A wide 150-450 floor is kept as a documented fallback
    when the reference does not carry 1-month effect sizes.
    """
    control = daily_steps.get("control", np.array([]))
    rl = daily_steps.get("rl", np.array([]))
    if control.size == 0 or rl.size == 0:
        return format_check_result(
            "T2.2",
            "Effect size magnitude",
            False,
            "Missing arm data",
            tier=2,
        )

    deltas = [
        float(v["delta"])
        for name, v in ref.get("effect_sizes", {}).items()
        if isinstance(v, dict) and "delta" in v and "1mo" in name
    ]
    if deltas:
        lo, hi = min(deltas), max(deltas)
        source = f"reference 1-month deltas [{lo:.0f}-{hi:.0f}]"
    else:
        lo, hi = 150.0, 450.0
        source = "fallback floor 150-450 (reference has no 1-month effect sizes)"

    ctrl_means = np.mean(control[:, :ONE_MONTH_DAYS], axis=1)
    rl_means = np.mean(rl[:, :ONE_MONTH_DAYS], axis=1)
    delta = float(np.mean(rl_means - ctrl_means))
    passed = lo <= delta <= hi
    detail = f"Δ={delta:.1f} steps (range: {source})"
    return format_check_result(
        "T2.2",
        "Effect size magnitude",
        passed,
        detail,
        tier=2,
    )


def check_t2_3_effect_size_ordering(
    daily_steps: dict[str, np.ndarray],
) -> dict:
    """T2.3: RL strictly beats Fixed, Random, and Control (gap map).

    The reference and r13 ladder show no reliable ordering between Fixed,
    Random, and Control — they cluster far below RL. The check therefore
    asserts RL > max(Fixed, Random, Control) without imposing a chain on the
    three non-RL arms.
    """
    arm_means: dict[str, float] = {}
    for arm in ARM_NAMES:
        data = daily_steps.get(arm, np.array([]))
        if data.size == 0:
            return format_check_result(
                "T2.3",
                "Effect size ordering",
                False,
                f"Missing arm: {arm}",
                tier=2,
            )
        arm_means[arm] = float(np.mean(data[:, :ONE_MONTH_DAYS]))

    non_rl_max = max(arm_means[arm] for arm in ("fixed", "random", "control"))
    passed = arm_means["rl"] > non_rl_max
    detail = (
        f"RL={arm_means['rl']:.1f} > max(Fixed, Random, Control)="
        f"{non_rl_max:.1f} (Fixed={arm_means['fixed']:.1f}, "
        f"Random={arm_means['random']:.1f}, Control={arm_means['control']:.1f})"
    )
    return format_check_result(
        "T2.3",
        "Effect size ordering",
        passed,
        detail,
        tier=2,
    )


def check_t2_4_attenuation_pattern(
    daily_steps: dict[str, np.ndarray],
) -> dict:
    """T2.4: 15% ≤ attenuation from 1mo to 2mo ≤ 45%."""
    control = daily_steps.get("control", np.array([]))
    rl = daily_steps.get("rl", np.array([]))
    if control.size == 0 or rl.size == 0:
        return format_check_result(
            "T2.4",
            "Attenuation pattern",
            False,
            "Missing arm data",
            tier=2,
        )

    ctrl_1mo = np.mean(control[:, :ONE_MONTH_DAYS], axis=1)
    rl_1mo = np.mean(rl[:, :ONE_MONTH_DAYS], axis=1)
    delta_1mo = float(np.mean(rl_1mo - ctrl_1mo))

    ctrl_2mo = np.mean(control[:, ONE_MONTH_DAYS:TWO_MONTH_DAYS], axis=1)
    rl_2mo = np.mean(rl[:, ONE_MONTH_DAYS:TWO_MONTH_DAYS], axis=1)
    delta_2mo = float(np.mean(rl_2mo - ctrl_2mo))

    if delta_1mo == 0:
        return format_check_result(
            "T2.4",
            "Attenuation pattern",
            False,
            "Zero 1-month effect, cannot compute attenuation",
            tier=2,
        )
    attenuation = (delta_1mo - delta_2mo) / delta_1mo
    passed = 0.15 <= attenuation <= 0.45
    detail = f"Δ₁={delta_1mo:.1f}, Δ₂={delta_2mo:.1f}, attenuation={attenuation:.1%}"
    return format_check_result(
        "T2.4",
        "Attenuation pattern",
        passed,
        detail,
        tier=2,
    )


def check_t2_5_between_person_variance(
    daily_steps: dict[str, np.ndarray],
) -> dict:
    """T2.5: ICC for daily steps: 0.4 ≤ ICC ≤ 0.9."""
    # Use control arm daily steps across seeds as "subjects"
    control = daily_steps.get("control", np.array([]))
    if control.size == 0:
        return format_check_result(
            "T2.5",
            "Between-person variance",
            False,
            "No control arm data",
            tier=2,
        )
    # Transpose: subjects (seeds) as rows, days as columns
    icc_val = icc_1way_random(control[:, :ONE_MONTH_DAYS])
    passed = 0.4 <= icc_val <= 0.9
    detail = f"ICC={icc_val:.4f} (range: 0.4–0.9)"
    return format_check_result(
        "T2.5",
        "Between-person variance",
        passed,
        detail,
        tier=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PEARL Constitution — Tier 2 Distribution Checks",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--persona", type=str, default="base", help="Persona name")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pearl_constitution_12action.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Dedicated stdout logger for the summary matrix
    results_logger = logging.getLogger(f"{__name__}.results")
    results_logger.propagate = False
    _stdout_handler = logging.StreamHandler(sys.stdout)
    _stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    results_logger.addHandler(_stdout_handler)
    results_logger.setLevel(logging.INFO)

    logger.info("Loading PEARL reference data...")
    ref = load_reference()
    ref_steps = ref["demographics"]["mean_baseline_steps"]

    logger.info("Loading config (persona=%s, seeds=%d)...", args.persona, args.seeds)
    config = load_constitution_config(args.persona, args.config)

    logger.info("Running all arms...")
    trajectories = run_all_arms(config, n_seeds=args.seeds)

    logger.info("Computing daily steps...")
    daily_steps = compute_arm_daily_steps(trajectories)

    logger.info("Running Tier 2 checks...")
    results = [
        check_t2_1_baseline_mean(daily_steps, ref_steps),
        check_t2_2_effect_size_magnitude(daily_steps, ref),
        check_t2_3_effect_size_ordering(daily_steps),
        check_t2_4_attenuation_pattern(daily_steps),
        check_t2_5_between_person_variance(daily_steps),
    ]

    matrix = format_matrix(results)
    results_logger.info(matrix)

    n_pass = sum(1 for r in results if r["passed"])
    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
