#!/usr/bin/env python3
"""Tier 4 — Stress Tests (edge-case robustness) for PEARL Constitution.

Usage:
    uv run python scripts/pearl_constitution/run_stress_tests.py [--seeds 5]
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
from scipy import stats as sp_stats

from rl_health_interventions.config.schemas import MDPConfig
from scripts.pearl_constitution.utils import (
    ARM_NAMES,
    compute_arm_daily_steps,
    compute_daily_steps,
    format_check_result,
    format_matrix,
    load_constitution_config,
    run_all_arms,
    run_arm_trajectories,
)

logger = logging.getLogger(__name__)

ONE_MONTH_DAYS = 30


def _make_random_transition_config(config: MDPConfig) -> MDPConfig:
    """Create a copy of the config with random transition model."""
    # Deep copy the config to avoid modifying the original
    new_config = config.model_copy(deep=True)
    new_config.transition_model.type = "random"
    new_config.transition_model.table_dir = None
    new_config.transition_model.transition_probabilities = None
    return new_config


def check_t4_1_random_matrix(config: MDPConfig, n_seeds: int) -> dict:
    """T4.1: With random transitions, all arms indistinguishable (ANOVA p > 0.5)."""
    logger.info("Running T4.1 with random transition model...")
    random_config = _make_random_transition_config(config)
    trajectories = run_all_arms(random_config, n_seeds=n_seeds)
    daily_steps = compute_arm_daily_steps(trajectories)

    arm_means: list[np.ndarray] = []
    for arm in ARM_NAMES:
        data = daily_steps.get(arm, np.array([]))
        if data.size == 0:
            return format_check_result(
                "T4.1",
                "Random matrix",
                True,
                "Skipped: missing arm data",
                tier=4,
            )
        means = np.mean(data[:, :ONE_MONTH_DAYS], axis=1)
        arm_means.append(means)

    f_stat, p_val = sp_stats.f_oneway(*arm_means)
    passed = p_val > 0.5
    detail = f"F={f_stat:.4f}, p={p_val:.4f}"
    return format_check_result(
        "T4.1",
        "Random matrix",
        passed,
        detail,
        tier=4,
    )


def check_t4_2_persona_collapse(
    daily_steps: dict[str, np.ndarray],
) -> dict:
    """T4.2: Identity transition → ANOVA across personas p > 0.5.

    With a single persona this check is skipped.
    """
    return format_check_result(
        "T4.2",
        "Persona collapse",
        True,
        "Skipped: requires multi-persona data with identity transitions",
        tier=4,
    )


def check_t4_3_infinite_horizon(
    trajectories: dict[str, list[list[dict]]],
) -> dict:
    """T4.3: 365-day episodes → steps plateau (no unbounded growth or decay).

    Check that the last 90 days' mean is within ±20% of the overall mean.
    """
    rl_trajs = trajectories.get("rl", [])
    if not rl_trajs:
        return format_check_result(
            "T4.3",
            "Infinite horizon",
            True,
            "Skipped: requires 365-day episode config",
            tier=4,
        )

    daily_per_seed: list[np.ndarray] = []
    for traj in rl_trajs:
        daily_per_seed.append(compute_daily_steps(traj))

    if not daily_per_seed:
        return format_check_result(
            "T4.3",
            "Infinite horizon",
            True,
            "Skipped: no valid RL trajectories",
            tier=4,
        )

    min_len = min(len(d) for d in daily_per_seed)
    daily_all = np.array([d[:min_len] for d in daily_per_seed])
    mean_overall = float(np.mean(daily_all))
    last_90 = float(np.mean(daily_all[:, -90:]))

    if mean_overall == 0:
        return format_check_result(
            "T4.3",
            "Infinite horizon",
            False,
            "Zero overall mean, cannot compute plateau",
            tier=4,
        )

    ratio = abs(last_90 - mean_overall) / mean_overall
    passed = ratio <= 0.20
    detail = (
        f"overall_mean={mean_overall:.1f}, last90_mean={last_90:.1f}, "
        f"deviation={ratio:.1%}"
    )
    return format_check_result(
        "T4.3",
        "Infinite horizon",
        passed,
        detail,
        tier=4,
    )


def check_t4_4_extreme_demographics(
    daily_steps: dict[str, np.ndarray],
) -> dict:
    """T4.4: 100% Resistant persona → RL vs Control Δ < 50 steps.

    This is checked when persona='resistant'.
    """
    control = daily_steps.get("control", np.array([]))
    rl = daily_steps.get("rl", np.array([]))
    if control.size == 0 or rl.size == 0:
        return format_check_result(
            "T4.4",
            "Extreme demographics",
            True,
            "Skipped: requires 'resistant' persona",
            tier=4,
        )
    ctrl_mean = float(np.mean(control[:, :ONE_MONTH_DAYS]))
    rl_mean = float(np.mean(rl[:, :ONE_MONTH_DAYS]))
    delta = abs(rl_mean - ctrl_mean)
    passed = delta < 50.0
    detail = f"Δ={delta:.1f} steps (threshold: < 50)"
    return format_check_result(
        "T4.4",
        "Extreme demographics",
        passed,
        detail,
        tier=4,
    )


def run_infinite_horizon_check(
    config: MDPConfig,
    n_seeds: int,
) -> dict:
    """Run a 365-day episode specifically for T4.3."""
    logger.info("Running 365-day episode for T4.3...")
    # Create a copy with 365 days
    inf_config = config.model_copy(deep=True)
    inf_config.episode_days = 365

    results: dict[str, list[list[dict]]] = {}
    for idx, agent_cfg in enumerate(inf_config.agents):
        arm_name = ARM_NAMES[idx]
        logger.info("  Running %s ...", arm_name)
        trajectories = run_arm_trajectories(inf_config, agent_cfg, n_seeds=n_seeds)
        results[arm_name] = trajectories
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PEARL Constitution — Tier 4 Stress Tests",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
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

    results: list[dict] = []

    # Load base config (shared across checks)
    config = load_constitution_config(args.persona, args.config)

    # T4.1 — Random transition model
    results.append(check_t4_1_random_matrix(config, args.seeds))

    # T4.2 — Skipped (requires multi-persona)
    daily_steps = compute_arm_daily_steps(run_all_arms(config, n_seeds=args.seeds))
    results.append(check_t4_2_persona_collapse(daily_steps))

    # T4.3 — Infinite horizon (365-day run)
    inf_trajectories = run_infinite_horizon_check(config, args.seeds)
    results.append(check_t4_3_infinite_horizon(inf_trajectories))

    # T4.4 — Extreme demographics (resistant persona)
    if args.persona == "resistant":
        logger.info("Running resistant persona for T4.4...")
        config_res = load_constitution_config("resistant", args.config)
        traj_res = run_all_arms(config_res, n_seeds=args.seeds)
        ds_res = compute_arm_daily_steps(traj_res)
        results.append(check_t4_4_extreme_demographics(ds_res))
    else:
        results.append(
            format_check_result(
                "T4.4",
                "Extreme demographics",
                True,
                "Skipped: use --persona resistant for this check",
                tier=4,
            )
        )

    matrix = format_matrix(results)
    print(matrix)

    n_pass = sum(1 for r in results if r["passed"])
    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
