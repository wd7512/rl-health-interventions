"""E2E benchmark: compare all agents on the sprint1 random-transition MDP.

Loads a sprint1_random MDP config with an agents section, runs every agent variant.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rl_health_interventions.config.loader import load_config
from _shared import agent_label, resolve_config, run_agent

logger = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

_SPRINT1_CONFIGS = [
    _CONFIGS_DIR / "sprint1_random.yaml",
    _CONFIGS_DIR / "sprint1_random_masked.yaml",
    _CONFIGS_DIR / "sprint1_random_extensions.yaml",
    _CONFIGS_DIR / "sprint1_random_extensions_masked.yaml",
]


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {n}")
    return n


def _benchmark_config(
    config_path: str,
    n_seeds: int,
    output_dir: Path | None = None,
    dump_json: bool = False,
    confirm_overwrite: bool = False,
) -> dict:
    config = load_config(config_path)
    config_name = Path(config_path).stem

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
            "total_reward": float(rewards.sum(axis=1).mean()),
            "total_std": float(rewards.sum(axis=1).std()),
            "per_step": float(rewards.mean()),
            "last50": float(rewards[:, -50:].mean()),
        }

    logger.info("\n%s", "=" * 72)
    logger.info("%-20s %14s %10s %10s", "Agent", "Total Reward", "Per Step", "Last 50")
    logger.info("%s", "-" * 72)
    for label in sorted(results, key=lambda k: results[k]["per_step"], reverse=True):
        r = results[label]
        logger.info(
            "%-20s %8.1f +- %-5.1f %9.4f %10.4f",
            label,
            r["total_reward"],
            r["total_std"],
            r["per_step"],
            r["last50"],
        )
    logger.info("%s", "=" * 72)

    if output_dir is not None and dump_json:
        _write_json_fixture(
            output_dir=output_dir,
            config_name=config_name,
            config_path=config_path,
            config_seed=config.seed,
            n_seeds=n_seeds,
            results=results,
            confirm_overwrite=confirm_overwrite,
        )

    return results


def _write_json_fixture(
    output_dir: Path,
    config_name: str,
    config_path: str,
    config_seed: int,
    n_seeds: int,
    results: dict[str, dict],
    confirm_overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{config_name}.json"

    if out_path.exists() and not confirm_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing fixture: {out_path}\n"
            f"Re-run with --confirm-overwrite to intentionally re-baseline."
        )

    resolved = Path(config_path).resolve()
    try:
        config_ref = str(resolved.relative_to(_CONFIGS_DIR.parent.parent.resolve()))
    except ValueError:
        config_ref = str(resolved)

    fixture = {
        "config": config_ref,
        "seed": config_seed,
        "seeds": n_seeds,
        "agents": results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    logger.info("Wrote fixture: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark agents on sprint1 random-transition MDP"
    )
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
        help="Config filename in configs/ (default: sprint1_random_extensions.yaml)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all sprint1_random configs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for JSON fixtures (use with --json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write JSON fixture files (requires --output)",
    )
    parser.add_argument(
        "--confirm-overwrite",
        action="store_true",
        help="Allow overwriting existing golden fixtures (use when re-baselining)",
    )
    args = parser.parse_args()

    if args.json and args.output is None:
        parser.error("--json requires --output <dir>")

    n_seeds = args.seeds
    output_dir = Path(args.output) if args.output else None

    if args.all:
        for config_path in _SPRINT1_CONFIGS:
            logger.info("\n=== Config: %s ===\n", config_path.name)
            _benchmark_config(
                str(config_path),
                n_seeds,
                output_dir=output_dir,
                dump_json=args.json,
                confirm_overwrite=args.confirm_overwrite,
            )
    else:
        _benchmark_config(
            resolve_config(args.config),
            n_seeds,
            output_dir=output_dir,
            dump_json=args.json,
            confirm_overwrite=args.confirm_overwrite,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
