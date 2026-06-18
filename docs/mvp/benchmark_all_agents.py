"""E2E benchmark: compare all contextual bandit agents (50 seeds each).

Loads one base MDP config, constructs agents programmatically.
Excluded from ruff/ty via pyproject.toml.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.experiment import run_episode
from rl_health_interventions.agents import derive_agent_seed, make as make_agent

N_SEEDS = 50
N_STEPS = 450  # 90 days * 5 steps

# Agent definitions: (label, config_type, kwargs)
AGENTS = [
    ("Standard TS", "thompson_sampling", {"alpha_prior": 1, "beta_prior": 1}),
    ("Contextual TS", "thompson_sampling", {"alpha_prior": 1, "beta_prior": 1, "contextual": True, "context_feature": "activity"}),
    ("Standard EG", "epsilon_greedy", {"epsilon": 0.1}),
    ("Contextual EG", "epsilon_greedy", {"epsilon": 0.1, "contextual": True, "context_feature": "activity"}),
    ("Standard UCB", "ucb", {"c": 2.0}),
    ("Contextual UCB", "ucb", {"c": 2.0, "contextual": True, "context_feature": "activity"}),
]


def run_seeds(agent_type: str, agent_kwargs: dict, config, n_seeds: int) -> np.ndarray:
    """Run agent over n_seeds, return per-step rewards (n_seeds, n_steps)."""
    all_rewards = []
    for seed in range(1, n_seeds + 1):
        kwargs = agent_kwargs.copy()
        kwargs["seed"] = derive_agent_seed(seed, agent_index=0)
        kwargs["actions"] = config.actions
        agent = make_agent(agent_type, **kwargs)
        df = run_episode(config, agent, seed=seed)
        all_rewards.append(df["reward"].values)
    return np.array(all_rewards)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(str(repo_root / "config" / "contextual_thompson.yaml"))

    results: dict[str, dict] = {}

    for label, agent_type, kwargs in AGENTS:
        print(f"Running {label} ({N_SEEDS} seeds)...")
        rewards = run_seeds(agent_type, kwargs, config, N_SEEDS)
        total_mean = rewards.sum(axis=1).mean()
        total_std = rewards.sum(axis=1).std()
        step_mean = rewards.mean()
        last50_mean = rewards[:, -50:].mean()
        results[label] = {
            "total_mean": total_mean,
            "total_std": total_std,
            "step_mean": step_mean,
            "last50_mean": last50_mean,
        }

    # Print summary table
    contextual_opt = 3 / 7  # 0.4286
    noncontextual_opt = 0.375

    print("\n" + "=" * 72)
    print(f"{'Agent':<20} {'Total Reward':>14} {'Per Step':>10} {'Last 50':>10} {'vs Ctx Opt':>12}")
    print("-" * 72)
    for label, _type, _kwargs in AGENTS:
        r = results[label]
        vs_opt = (r["step_mean"] - contextual_opt) / contextual_opt * 100
        print(f"{label:<20} {r['total_mean']:>8.1f} ± {r['total_std']:<5.1f} {r['step_mean']:>9.4f} {r['last50_mean']:>10.4f} {vs_opt:>+10.1f}%")
    print("-" * 72)
    print(f"{'Contextual optimal':<20} {'192.9':>14} {'0.4286':>10} {'0.4286':>10} {'---':>12}")
    print(f"{'Non-ctx optimal':<20} {'168.8':>14} {'0.3750':>10} {'0.3750':>10} {'---':>12}")
    print("=" * 72)


if __name__ == "__main__":
    main()
