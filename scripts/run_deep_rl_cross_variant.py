#!/usr/bin/env python3
"""Run deep RL benchmarks across all Sprint 1 variant configs.

This script reproduces the cross-variant deep RL benchmark that was lost during
the PR #239 refactor. It runs Q-Learning, DQN, REINFORCE, and PPO on all
Sprint 1 random and bootstrap variants.

Usage:
    python scripts/run_deep_rl_cross_variant.py

Output:
    Prints LaTeX-formatted table with results for each config.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from rl_health_interventions.sweep import run_experiment_multi_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Deep RL agent configs (same as sprint1_bootstrap_deep_rl.yaml)
DEEP_RL_AGENTS = [
    {
        "type": "q_learning",
        "lr": 0.1,
        "gamma": 0.95,
        "epsilon": 0.1,
    },
    {
        "type": "dqn",
        "lr": 0.01,
        "gamma": 0.95,
        "epsilon": 0.2,
        "batch_size": 8,
        "buffer_size": 64,
        "target_update_freq": 10,
        "grad_clip": 1.0,
        "hidden_dim": [16, 8],
        "state_dim": 32,
    },
    {
        "type": "reinforce",
        "lr": 0.01,
        "gamma": 0.99,
        "hidden_dim": [16, 8],
        "state_dim": 32,
    },
    {
        "type": "ppo",
        "lr": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ppo_epochs": 2,
        "hidden_dim": [16, 8],
        "state_dim": 32,
    },
]

# All configs to run
CONFIGS = [
    # Random variants
    "docs/experimental_phases/sprint1_random/configs/sprint1_random.yaml",
    "docs/experimental_phases/sprint1_random/configs/sprint1_random_masked.yaml",
    "docs/experimental_phases/sprint1_random/configs/sprint1_random_extensions.yaml",
    "docs/experimental_phases/sprint1_random/configs/sprint1_random_extensions_masked.yaml",
    # Bootstrap variants
    "docs/experimental_phases/sprint1_bootstrap/configs/sprint1_bootstrap.yaml",
    "docs/experimental_phases/sprint1_bootstrap/configs/sprint1_bootstrap_masked.yaml",
    "docs/experimental_phases/sprint1_bootstrap/configs/sprint1_bootstrap_extensions.yaml",
    "docs/experimental_phases/sprint1_bootstrap/configs/sprint1_bootstrap_extensions_masked.yaml",
    "docs/experimental_phases/sprint1_bootstrap/configs/sprint1_bootstrap_context_burden.yaml",
]


def load_config_with_agents(config_path: str) -> dict[str, Any]:
    """Load a config YAML and inject deep RL agents."""
    path = Path(config_path)
    if not path.exists():
        msg = f"Config not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        config = yaml.safe_load(f)

    # Add deep RL agents
    config["agents"] = DEEP_RL_AGENTS.copy()
    return config


def run_benchmark(
    config_path: str,
    n_seeds: int = 50,
    output_json: str | None = None,
) -> dict[str, dict[str, float]]:
    """Run deep RL benchmark on a single config."""
    logger.info("Running benchmark on %s", config_path)

    base_config = load_config_with_agents(config_path)
    config_dir = Path(config_path).parent.resolve()

    # Fix relative paths in transition_model for bootstrap configs
    if base_config.get("transition_model", {}).get("type") == "bootstrap":
        table_dir = base_config["transition_model"].get("table_dir", "")
        if not Path(table_dir).is_absolute():
            # Convert relative path to absolute from config file location
            abs_table_dir = (config_dir / table_dir).resolve()
            base_config["transition_model"]["table_dir"] = str(abs_table_dir)
            logger.info("Fixed table_dir to: %s", abs_table_dir)

    # Write to temp file in the config's directory so relative paths work
    tmp_path = config_dir / f"temp_deep_rl_{Path(config_path).stem}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(base_config, f)

    try:
        results = run_experiment_multi_seed(str(tmp_path), n_seeds=n_seeds)
    finally:
        tmp_path.unlink(missing_ok=True)

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote results to %s", output_json)

    return results


def main() -> int:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser(description="Run deep RL cross-variant benchmarks")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=50,
        help="Number of seeds per agent (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/deep_rl_cross_variant"),
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    all_results: dict[str, dict[str, Any]] = {}

    for config in CONFIGS:
        config_name = Path(config).stem
        output_json = args.output_dir / f"{config_name}.json"

        try:
            results = run_benchmark(
                config,
                n_seeds=args.n_seeds,
                output_json=str(output_json),
            )
            all_results[config_name] = results
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("Failed on %s: %s", config, e)
            all_results[config_name] = {"error": str(e)}  # type: ignore[assignment]

    # Print LaTeX table
    print("\n" + "=" * 80)
    print("Deep RL Cross-Variant Results (50 seeds)")
    print("=" * 80)

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("Config & DQN & PPO & Q-Learning & REINFORCE \\\\")
    print("\\midrule")

    for config_name, results in all_results.items():
        if "error" in results:
            print(f"{config_name} & -- & -- & -- & -- \\\\")
            continue

        dqn = results.get("dqn", {}).get("total_reward", 0)
        dqn_std = results.get("dqn", {}).get("total_std", 0)
        ppo = results.get("ppo", {}).get("total_reward", 0)
        ppo_std = results.get("ppo", {}).get("total_std", 0)
        ql = results.get("q_learning", {}).get("total_reward", 0)
        ql_std = results.get("q_learning", {}).get("total_std", 0)
        rf = results.get("reinforce", {}).get("total_reward", 0)
        rf_std = results.get("reinforce", {}).get("total_std", 0)

        print(
            f"{config_name} & "
            f"${dqn:.1f} \\pm {dqn_std:.1f}$ & "
            f"${ppo:.1f} \\pm {ppo_std:.1f}$ & "
            f"${ql:.1f} \\pm {ql_std:.1f}$ & "
            f"${rf:.1f} \\pm {rf_std:.1f}$ \\\\"
        )

    print("\\bottomrule")
    print("\\end{tabular}")
    print(
        "\\caption{Deep RL results across Sprint 1 variants (mean total reward "
        "\\pm SD, 50 seeds)}"
    )
    print("\\label{tab:deep_rl_cross_variant}")
    print("\\end{table}")

    # Save summary JSON
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
