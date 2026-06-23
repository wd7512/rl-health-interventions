"""E2E benchmark: compare all contextual bandit agents.

Loads a base MDP config with an agents section, runs every agent variant.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


from rl_health_interventions.config.loader import load_config
from _shared import agent_label, resolve_config, run_agent

logger = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

_MVP_CONFIGS = [
    _CONFIGS_DIR / "mvp.yaml",
    _CONFIGS_DIR / "mvp_masked.yaml",
    _CONFIGS_DIR / "mvp_extensions.yaml",
    _CONFIGS_DIR / "mvp_extensions_masked.yaml",
]


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {n}")
    return n


def _benchmark_config(config_path: str, n_seeds: int) -> None:
    config = load_config(config_path)

    n_steps = config.episode_days * config.steps_per_day

    logger.info("Config: %s", config_path)
    logger.info(
        "MDP: %d days x %d steps = %d steps",
        config.episode_days,
        config.steps_per_day,
        n_steps,
    )
    logger.info("Seeds: %d\n", n_seeds)

    results: dict[str, dict] = {}
    for agent_cfg in config.agents:
        label = agent_label(agent_cfg)
        logger.info("Running %s...", label)
        rewards = run_agent(config, agent_cfg, n_seeds)
        results[label] = {
            "total_mean": rewards.sum(axis=1).mean(),
            "total_std": rewards.sum(axis=1).std(),
            "step_mean": rewards.mean(),
            "last50_mean": rewards[:, -50:].mean(),
        }

    logger.info("\n%s", "=" * 72)
    logger.info("%-20s %14s %10s %10s", "Agent", "Total Reward", "Per Step", "Last 50")
    logger.info("%s", "-" * 72)
    for label in sorted(results, key=lambda k: results[k]["step_mean"], reverse=True):
        r = results[label]
        logger.info(
            "%-20s %8.1f +- %-5.1f %9.4f %10.4f",
            label,
            r["total_mean"],
            r["total_std"],
            r["step_mean"],
            r["last50_mean"],
        )
    logger.info("%s", "=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark contextual bandit agents")
    parser.add_argument(
        "--seeds",
        type=_positive_int,
        default=50,
        help="Number of random seeds (default: 50)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config filename in configs/ (default: mvp_extensions.yaml)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all MVP configs",
    )
    args = parser.parse_args()

    n_seeds = args.seeds

    if args.all:
        for config_path in _MVP_CONFIGS:
            logger.info("\n=== Config: %s ===\n", config_path.name)
            _benchmark_config(str(config_path), n_seeds)
    else:
        _benchmark_config(resolve_config(args.config), n_seeds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
