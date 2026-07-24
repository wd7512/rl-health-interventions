"""E2E benchmark: compare PEARL-matched arms on random-transition MDP."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from _shared import agent_label, resolve_config, run_agent

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.evaluation.metrics import compute_metrics, format_comparison_table

logger = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
_PEARL_CONFIGS = [_CONFIGS_DIR / "pearl_random.yaml"]


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
    for i, agent_cfg in enumerate(config.agents):
        label = agent_label(agent_cfg)
        logger.info("Running %s...", label)
        rewards = run_agent(config, agent_cfg, n_seeds, agent_index=i)
        results[label] = compute_metrics(rewards)

    logger.info("\n%s", format_comparison_table(results))

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
            "Re-run with --confirm-overwrite to intentionally re-baseline."
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
        description="Benchmark PEARL 4-arm setup on random-transition MDP"
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
        help="Config filename in configs/ (default: pearl_random.yaml)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all pearl_random configs",
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
        for config_path in _PEARL_CONFIGS:
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
