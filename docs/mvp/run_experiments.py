"""E2E benchmark: compare all contextual bandit agents.

Loads a base MDP config with an agents section, runs every agent variant.
Reveals how ergonomic the experiment API is for real usage.
"""
from __future__ import annotations

import argparse
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import AgentConfig
from rl_health_interventions.experiment import run_episode
from rl_health_interventions.agents import derive_agent_seed, make as make_agent

_AGENT_SHORT_NAMES: dict[str, str] = {
    "thompson_sampling": "TS",
    "epsilon_greedy": "EG",
    "ucb": "UCB",
    "decaying_epsilon_greedy": "DEC",
    "random": "Random",
}


def _agent_label(cfg: AgentConfig) -> str:
    if cfg.type == "random":
        return "Random"
    short = _AGENT_SHORT_NAMES.get(cfg.type, cfg.type)
    prefix = "Contextual" if cfg.contextual else "Standard"
    return f"{prefix} {short}"


def run_agent(config, agent_cfg: AgentConfig, n_seeds: int) -> np.ndarray:
    """Run one agent variant over n_seeds. Returns per-step rewards (n_seeds, n_steps)."""
    skip_keys = {"type"}
    if not agent_cfg.contextual:
        skip_keys.update({"contextual", "context_feature"})
    rewards = []
    for seed in range(1, n_seeds + 1):
        kwargs = {
            k: v
            for k, v in agent_cfg.model_dump().items()
            if v is not None and k not in skip_keys
        }
        kwargs["actions"] = config.actions
        kwargs["seed"] = derive_agent_seed(seed, agent_index=0)
        agent = make_agent(agent_cfg.type, **kwargs)
        df = run_episode(config, agent, seed=seed)
        rewards.append(df["reward"].values)
    return np.array(rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark contextual bandit agents")
    parser.add_argument(
        "--seeds",
        type=int,
        default=50,
        help="Number of random seeds (default: 50)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Base MDP config (default: docs/mvp/configs/mvp.yaml)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = args.config or str(
        repo_root / "docs" / "mvp" / "configs" / "mvp.yaml"
    )
    config = load_config(config_path)

    n_steps = config.episode_days * config.steps_per_day
    n_seeds = args.seeds

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
        label = _agent_label(agent_cfg)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
