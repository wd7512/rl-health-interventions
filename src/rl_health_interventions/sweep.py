from __future__ import annotations

import csv
import logging
from pathlib import Path

from rl_health_interventions.agents import derive_agent_seed
from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.episode import run_episode
from rl_health_interventions.evaluation._shared import run_agent
from rl_health_interventions.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def run_experiment(config_path: str | Path) -> dict[str, float]:
    """Run all agents defined in the config, return {agent_type: total_reward}."""
    config = load_config(config_path)
    results: dict[str, float] = {}
    for i, agent_cfg in enumerate(config.agents):
        kwargs = {
            k: v
            for k, v in agent_cfg.model_dump().items()
            if v is not None and k != "type"
        }
        kwargs["actions"] = config.action_names
        kwargs["seed"] = derive_agent_seed(config.seed, agent_index=i)
        agent = make_agent(agent_cfg.type, **kwargs)
        records = run_episode(config, agent, seed=config.seed)
        key = agent_cfg.type
        suffix = 1
        while key in results:
            key = f"{agent_cfg.type}_{suffix}"
            suffix += 1
        results[key] = sum(r["reward"] for r in records)
    return results


def run_experiment_multi_seed(
    config_path: str | Path,
    n_seeds: int = 50,
) -> dict[str, dict[str, float]]:
    """Run all agents over n_seeds, return {agent_type: metrics_dict}.

    Metrics: total_reward, total_std, per_step, last50.
    """
    config = load_config(config_path)
    logger.info("Config: %s | seeds: %d", config_path, n_seeds)
    results: dict[str, dict[str, float]] = {}
    for agent_cfg in config.agents:
        logger.info("Running %s...", agent_cfg.type)
        rewards = run_agent(config, agent_cfg, n_seeds)
        metrics = compute_metrics(rewards)
        key = agent_cfg.type
        suffix = 1
        while key in results:
            key = f"{agent_cfg.type}_{suffix}"
            suffix += 1
        results[key] = metrics
        logger.info(
            "  %s: total_reward=%.1f, per_step=%.4f, last50=%.4f",
            key,
            metrics["total_reward"],
            metrics["per_step"],
            metrics["last50"],
        )
    return results


def run_experiment_csv(
    config_path: str | Path,
    n_seeds: int,  # noqa: ARG001
    output_csv: Path,
) -> None:
    """Run all agents and write per-step CSV with factored state dimensions.

    Columns: step, day, step_of_day, state_factors..., action, reward, agent_type.
    """
    config = load_config(config_path)
    all_records: list[dict] = []
    for i, agent_cfg in enumerate(config.agents):
        exclude: set[str] = {"type"}
        if not getattr(agent_cfg, "contextual", False):
            exclude |= {"contextual", "context_features"}
        kwargs = agent_cfg.model_dump(exclude=exclude, exclude_none=True)
        kwargs["actions"] = config.action_names
        kwargs["seed"] = derive_agent_seed(config.seed, agent_index=i)
        agent = make_agent(agent_cfg.type, **kwargs)
        records = run_episode(config, agent, seed=config.seed)
        for r in records:
            r["agent_type"] = agent_cfg.type
        all_records.extend(records)
        logger.info("  %s: %d steps", agent_cfg.type, len(records))

    if not all_records:
        logger.warning("No records to write to CSV")
        return

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_records[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    logger.info("Wrote %d rows to %s", len(all_records), output_path)
