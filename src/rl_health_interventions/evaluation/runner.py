"""E2E benchmark runner: compare all agents defined in a config."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.evaluation._shared import agent_label, run_agent
from rl_health_interventions.evaluation.metrics import (
    compute_metrics,
    format_comparison_table,
)

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    """Argparse type for positive integers."""
    n = int(value)
    if n <= 0:
        msg = f"must be a positive integer, got {n}"
        raise argparse.ArgumentTypeError(msg)
    return n


def run_benchmark(
    config_path: str,
    n_seeds: int,
    output_dir: Path | None = None,
    dump_json: bool = False,
    confirm_overwrite: bool = False,
    config_ref_base: Path | None = None,
) -> dict[str, dict]:
    """Run all agents defined in *config_path* over *n_seeds*.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    n_seeds : int
        Number of random seeds per agent.
    output_dir : Path or None
        Directory for JSON fixture output (used when *dump_json* is True).
    dump_json : bool
        Whether to write a JSON fixture file.
    confirm_overwrite : bool
        Allow overwriting an existing fixture file.
    config_ref_base : Path or None
        Base path for computing the relative ``config`` value stored in the
        fixture.  When ``None`` the resolved absolute path is used.

    Returns
    -------
    dict[str, dict]
        Mapping from agent label to metrics dict with keys ``total_reward``,
        ``total_std``, ``per_step``, ``last50``.
    """
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
        results[label] = compute_metrics(rewards)

    table = format_comparison_table(results)
    logger.info("\n%s", table)

    if output_dir is not None and dump_json:
        _write_json_fixture(
            output_dir=output_dir,
            config_name=config_name,
            config_path=config_path,
            config_seed=config.seed,
            n_seeds=n_seeds,
            results=results,
            confirm_overwrite=confirm_overwrite,
            config_ref_base=config_ref_base,
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
    config_ref_base: Path | None = None,
) -> None:
    """Write a JSON fixture file for the benchmark results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{config_name}.json"

    if out_path.exists() and not confirm_overwrite:
        msg = (
            f"Refusing to overwrite existing fixture: {out_path}\n"
            "Re-run with --confirm-overwrite to intentionally re-baseline."
        )
        raise FileExistsError(msg)

    resolved = Path(config_path).resolve()
    if config_ref_base is not None:
        try:
            config_ref = str(resolved.relative_to(config_ref_base.resolve()))
        except ValueError:
            config_ref = str(resolved)
    else:
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
