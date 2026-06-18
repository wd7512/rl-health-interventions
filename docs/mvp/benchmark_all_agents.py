"""E2E benchmark: compare all contextual bandit agents.

Loads a base MDP config, swaps out the agent section for each variant.
Reveals how ergonomic the experiment API is for real usage.
"""
from __future__ import annotations

import argparse
import copy
import numpy as np
from pathlib import Path

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import AgentConfig
from rl_health_interventions.experiment import run_episode
from rl_health_interventions.agents import derive_agent_seed, make as make_agent

# Agent variants to benchmark: (label, agent_config_dict)
AGENT_VARIANTS = [
    ("Standard TS", {"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1}),
    ("Contextual TS", {"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1, "contextual": True, "context_feature": "activity"}),
    ("Standard EG", {"type": "epsilon_greedy", "epsilon": 0.1}),
    ("Contextual EG", {"type": "epsilon_greedy", "epsilon": 0.1, "contextual": True, "context_feature": "activity"}),
    ("Standard UCB", {"type": "ucb", "c": 2.0}),
    ("Contextual UCB", {"type": "ucb", "c": 2.0, "contextual": True, "context_feature": "activity"}),
]


def run_agent(config, agent_cfg: AgentConfig, n_seeds: int) -> np.ndarray:
    """Run one agent variant over n_seeds. Returns per-step rewards (n_seeds, n_steps)."""
    rewards = []
    for seed in range(1, n_seeds + 1):
        kwargs = {k: v for k, v in agent_cfg.model_dump().items() if v is not None and k != "type"}
        kwargs["actions"] = config.actions
        kwargs["seed"] = derive_agent_seed(seed, agent_index=0)
        agent = make_agent(agent_cfg.type, **kwargs)
        df = run_episode(config, agent, seed=seed)
        rewards.append(df["reward"].values)
    return np.array(rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark contextual bandit agents")
    parser.add_argument("--seeds", type=int, default=50, help="Number of random seeds (default: 50)")
    parser.add_argument("--config", type=str, default=None, help="Base MDP config (default: config/rule_based.yaml)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = args.config or str(repo_root / "config" / "rule_based.yaml")
    config = load_config(config_path)

    n_steps = config.episode_days * config.steps_per_day
    n_seeds = args.seeds

    print(f"Config: {config_path}")
    print(f"MDP: {config.episode_days} days × {config.steps_per_day} steps = {n_steps} steps")
    print(f"Seeds: {n_seeds}\n")

    results: dict[str, dict] = {}
    for label, agent_dict in AGENT_VARIANTS:
        agent_cfg = AgentConfig.model_validate(agent_dict)
        print(f"Running {label}...")
        rewards = run_agent(config, agent_cfg, n_seeds)
        results[label] = {
            "total_mean": rewards.sum(axis=1).mean(),
            "total_std": rewards.sum(axis=1).std(),
            "step_mean": rewards.mean(),
            "last50_mean": rewards[:, -50:].mean(),
        }

    contextual_opt = 3 / 7
    print(f"\n{'='*72}")
    print(f"{'Agent':<20} {'Total Reward':>14} {'Per Step':>10} {'Last 50':>10} {'vs Ctx Opt':>12}")
    print(f"{'-'*72}")
    for label, _ in AGENT_VARIANTS:
        r = results[label]
        vs_opt = (r["step_mean"] - contextual_opt) / contextual_opt * 100
        print(f"{label:<20} {r['total_mean']:>8.1f} ± {r['total_std']:<5.1f} {r['step_mean']:>9.4f} {r['last50_mean']:>10.4f} {vs_opt:>+10.1f}%")
    print(f"{'-'*72}")
    print(f"{'Contextual optimal':<20} {192.9:>14} {0.4286:>10} {0.4286:>10} {'---':>12}")
    print(f"{'Non-ctx optimal':<20} {168.8:>14} {0.3750:>10} {0.3750:>10} {'---':>12}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
