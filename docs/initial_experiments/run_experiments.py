"""E2E benchmark: compare all contextual bandit agents.

Loads a base MDP config, swaps out the agent section for each variant.
Reveals how ergonomic the experiment API is for real usage.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import AgentConfig

from _experiment_utils import run_agent

# Agent variants to benchmark: (label, agent_config_dict)
AGENT_VARIANTS = [
    ("Standard TS", {"type": "thompson_sampling", "alpha_prior": 1.0, "beta_prior": 1.0}),
    ("Contextual TS", {"type": "thompson_sampling", "alpha_prior": 1.0, "beta_prior": 1.0, "contextual": True, "context_feature": "activity"}),
    ("Standard EG", {"type": "epsilon_greedy", "epsilon": 0.2}),
    ("Contextual EG", {"type": "epsilon_greedy", "epsilon": 0.1, "contextual": True, "context_feature": "activity"}),
    ("Standard UCB", {"type": "ucb", "c": 0.5}),
    ("Contextual UCB", {"type": "ucb", "c": 0.2, "contextual": True, "context_feature": "activity"}),
    ("Standard DEC", {"type": "decaying_epsilon_greedy", "epsilon_start": 0.8, "epsilon_min": 0.01, "decay_steps": 75}),
    ("Contextual DEC", {"type": "decaying_epsilon_greedy", "epsilon_start": 0.3, "epsilon_min": 0.01, "decay_steps": 300, "contextual": True, "context_feature": "activity"}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark contextual bandit agents")
    parser.add_argument("--seeds", type=int, default=50, help="Number of random seeds (default: 50)")
    parser.add_argument("--config", type=str, default=None, help="Base MDP config (default: config/rule_based.yaml)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = args.config or str(repo_root / "docs" / "initial_experiments" / "configs" / "rule_based_with_actions.yaml")
    config = load_config(config_path)

    n_steps = config.episode_days * config.steps_per_day
    n_seeds = args.seeds

    logger.info("Config: %s", config_path)
    logger.info("MDP: %d days × %d steps = %d steps", config.episode_days, config.steps_per_day, n_steps)
    logger.info("Seeds: %d\n", n_seeds)

    results: dict[str, dict] = {}
    for label, agent_dict in AGENT_VARIANTS:
        agent_cfg = AgentConfig.model_validate(agent_dict)
        logger.info("Running %s...", label)
        rewards = run_agent(config, agent_cfg, n_seeds)
        results[label] = {
            "total_mean": rewards.sum(axis=1).mean(),
            "total_std": rewards.sum(axis=1).std(),
            "step_mean": rewards.mean(),
            "last50_mean": rewards[:, -50:].mean(),
        }

    contextual_opt = 0.75
    contextual_total_opt = contextual_opt * n_steps
    noncontextual_opt = 5 / 9
    noncontextual_total_opt = noncontextual_opt * n_steps
    logger.info("\n%s", "=" * 72)
    logger.info("%-20s %14s %10s %10s %12s %12s", "Agent", "Total Reward", "Per Step", "Last 50", "vs Ctx Opt", "vs Non-Ctx Opt")
    logger.info("%s", "-" * 72)
    for label, _ in AGENT_VARIANTS:
        r = results[label]
        vs_ctx = (r["step_mean"] - contextual_opt) / contextual_opt * 100
        vs_nctx = (r["step_mean"] - noncontextual_opt) / noncontextual_opt * 100
        logger.info("%-20s %8.1f ± %-5.1f %9.4f %10.4f %+10.1f%% %+10.1f%%", label, r["total_mean"], r["total_std"], r["step_mean"], r["last50_mean"], vs_ctx, vs_nctx)
    logger.info("%s", "-" * 72)
    logger.info("%-20s %14.1f %10.4f %10.4f %12s %12s", "Contextual optimal", contextual_total_opt, contextual_opt, contextual_opt, "---", "+34.9%")
    logger.info("%-20s %14.1f %10.4f %10.4f %12s %12s", "Non-ctx optimal", noncontextual_total_opt, noncontextual_opt, noncontextual_opt, "-25.9%", "---")
    logger.info("%s", "=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
