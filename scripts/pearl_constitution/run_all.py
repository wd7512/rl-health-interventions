#!/usr/bin/env python3
"""Master runner for all PEARL Constitution validation tiers.

Usage:
    uv run python scripts/pearl_constitution/run_all.py --seeds 10
    uv run python scripts/pearl_constitution/run_all.py --seeds 5 --all-personas
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
from scipy import stats as sp_stats

from scripts.pearl_constitution.run_baseline_check import (
    check_t1_1_baseline_stability,
    check_t1_2_action_differentiation,
    check_t1_3_direction_correctness,
    check_t1_4_no_degenerate_trajectories,
)
from scripts.pearl_constitution.run_behaviour_check import (
    check_t3_1_burden_saturation,
    check_t3_2_persona_heterogeneity,
    check_t3_3_weekend_effect,
    check_t3_4_non_response_detection,
)
from scripts.pearl_constitution.run_distribution_check import (
    check_t2_1_baseline_mean,
    check_t2_2_effect_size_magnitude,
    check_t2_3_effect_size_ordering,
    check_t2_4_attenuation_pattern,
    check_t2_5_between_person_variance,
)
from scripts.pearl_constitution.run_stress_tests import (
    check_t4_1_random_matrix,
    check_t4_2_persona_collapse,
    check_t4_3_infinite_horizon,
    check_t4_4_extreme_demographics,
    run_infinite_horizon_check,
)
from scripts.pearl_constitution.utils import (
    PERSONA_TABLE_DIRS,
    compute_arm_daily_steps,
    format_check_result,
    format_matrix,
    load_constitution_config,
    load_reference,
    run_all_arms,
)

logger = logging.getLogger(__name__)


def run_tier_1(config, n_seeds: int, ref_steps: float) -> list[dict]:
    """Run all Tier 1 checks."""
    logger.info("=== Tier 1: Sanity Checks ===")
    trajectories = run_all_arms(config, n_seeds=n_seeds)
    daily_steps = compute_arm_daily_steps(trajectories)
    return [
        check_t1_1_baseline_stability(daily_steps, ref_steps),
        check_t1_2_action_differentiation(daily_steps),
        check_t1_3_direction_correctness(daily_steps),
        check_t1_4_no_degenerate_trajectories(daily_steps),
    ]


def run_tier_2(config, n_seeds: int, ref_steps: float) -> list[dict]:
    """Run all Tier 2 checks."""
    logger.info("=== Tier 2: Distribution Checks ===")
    trajectories = run_all_arms(config, n_seeds=n_seeds)
    daily_steps = compute_arm_daily_steps(trajectories)
    return [
        check_t2_1_baseline_mean(daily_steps, ref_steps),
        check_t2_2_effect_size_magnitude(daily_steps),
        check_t2_3_effect_size_ordering(daily_steps),
        check_t2_4_attenuation_pattern(daily_steps),
        check_t2_5_between_person_variance(daily_steps),
    ]


def run_tier_3(config, n_seeds: int, ref_steps: float) -> list[dict]:
    """Run all Tier 3 checks."""
    logger.info("=== Tier 3: Behavioural Checks ===")
    trajectories = run_all_arms(config, n_seeds=n_seeds)
    daily_steps = compute_arm_daily_steps(trajectories)

    return [
        check_t3_1_burden_saturation(daily_steps),
        check_t3_2_persona_heterogeneity(daily_steps),
        check_t3_3_weekend_effect(trajectories),
        check_t3_4_non_response_detection(daily_steps),
    ]


def run_tier_4(
    config, n_seeds: int, ref_steps: float, persona: str = "base"
) -> list[dict]:
    """Run all Tier 4 checks."""
    logger.info("=== Tier 4: Stress Tests ===")
    results: list[dict] = []

    # T4.1 — Random transition
    results.append(check_t4_1_random_matrix(config, n_seeds))

    # T4.2 — Skipped
    trajectories = run_all_arms(config, n_seeds=n_seeds)
    daily_steps = compute_arm_daily_steps(trajectories)
    results.append(check_t4_2_persona_collapse(daily_steps))

    # T4.3 — Infinite horizon
    inf_trajectories = run_infinite_horizon_check(config, n_seeds)
    results.append(check_t4_3_infinite_horizon(inf_trajectories))

    # T4.4 — Resistant persona
    if persona == "resistant":
        config_res = load_constitution_config("resistant")
        traj_res = run_all_arms(config_res, n_seeds=n_seeds)
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

    return results


def check_t3_2_across_personas(
    persona_results: dict[str, dict[str, np.ndarray]],
) -> dict:
    """Compute T3.2 across multiple personas."""
    groups: list[np.ndarray] = []
    persona_labels: list[str] = []
    for persona, arm_data in persona_results.items():
        rl = arm_data.get("rl", np.array([]))
        control = arm_data.get("control", np.array([]))
        if rl.size > 0 and control.size > 0:
            effect = np.mean(rl[:, :30], axis=1) - np.mean(control[:, :30], axis=1)
            groups.append(effect)
            persona_labels.append(persona)

    if len(groups) < 2:
        return format_check_result(
            "T3.2",
            "Persona heterogeneity",
            True,
            "Skipped: requires data from 2+ personas",
            tier=3,
        )

    f_stat, p_val = sp_stats.f_oneway(*groups)
    # Eta-squared
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_total = float(np.sum((all_data - grand_mean) ** 2))
    ss_between = sum(len(g) * float(np.mean(g) - grand_mean) ** 2 for g in groups)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    passed = p_val < 0.01 and eta_sq > 0.1
    detail = (
        f"F={f_stat:.4f}, p={p_val:.6f}, η²={eta_sq:.4f} "
        f"(personas: {', '.join(persona_labels)})"
    )
    return format_check_result("T3.2", "Persona heterogeneity", passed, detail, tier=3)


def run_all_personas(
    n_seeds: int,
    config_path: str = "config/pearl_constitution.yaml",
) -> dict[str, dict[str, np.ndarray]]:
    """Run all arms across all personas."""
    persona_results: dict[str, dict[str, np.ndarray]] = {}
    for persona in PERSONA_TABLE_DIRS:
        logger.info("Running persona: %s", persona)
        config = load_constitution_config(persona, config_path)
        trajectories = run_all_arms(config, n_seeds=n_seeds)
        daily_steps = compute_arm_daily_steps(trajectories)
        persona_results[persona] = daily_steps
    return persona_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PEARL Constitution — Master Validation Runner",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--persona", type=str, default="base", help="Persona name")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pearl_constitution.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--all-personas",
        action="store_true",
        help="Run across all personas (for T3.2)",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default="1,2,3,4",
        help="Comma-separated tiers to run (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    ref = load_reference()
    ref_steps = ref["demographics"]["mean_baseline_steps"]

    tiers_to_run = {int(t.strip()) for t in args.tiers.split(",")}
    all_results: list[dict] = []

    config = load_constitution_config(args.persona, args.config)

    if 1 in tiers_to_run:
        all_results.extend(run_tier_1(config, args.seeds, ref_steps))
    if 2 in tiers_to_run:
        all_results.extend(run_tier_2(config, args.seeds, ref_steps))
    if 3 in tiers_to_run:
        all_results.extend(run_tier_3(config, args.seeds, ref_steps))
    if 4 in tiers_to_run:
        all_results.extend(run_tier_4(config, args.seeds, ref_steps, args.persona))

    # Run T3.2 across all personas if requested
    if args.all_personas:
        logger.info("Running multi-persona analysis for T3.2...")
        persona_results = run_all_personas(args.seeds, args.config)
        t3_2_result = check_t3_2_across_personas(persona_results)
        # Replace the placeholder T3.2 result with the real one
        all_results = [r for r in all_results if r["check_id"] != "T3.2"]
        all_results.append(t3_2_result)

    matrix = format_matrix(all_results)
    print(matrix)

    n_pass = sum(1 for r in all_results if r["passed"])
    n_total = len(all_results)
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
