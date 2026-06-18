"""Generate learning curve charts for all contextual bandit agents.

Loads a base MDP config, swaps out the agent section for each variant.
Excluded from ty/ruff via pyproject.toml.
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import AgentConfig
from rl_health_interventions.experiment import run_episode
from rl_health_interventions.agents import derive_agent_seed, make as make_agent

AGENT_VARIANTS = [
    ("Standard TS", {"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1}, "#2196F3", "-"),
    ("Contextual TS", {"type": "thompson_sampling", "alpha_prior": 1, "beta_prior": 1, "contextual": True, "context_feature": "activity"}, "#4CAF50", "-"),
    ("Standard EG", {"type": "epsilon_greedy", "epsilon": 0.1}, "#FF9800", "-"),
    ("Contextual EG", {"type": "epsilon_greedy", "epsilon": 0.1, "contextual": True, "context_feature": "activity"}, "#E91E63", "-"),
    ("Standard UCB", {"type": "ucb", "c": 2.0}, "#9C27B0", "-"),
    ("Contextual UCB", {"type": "ucb", "c": 2.0, "contextual": True, "context_feature": "activity"}, "#00BCD4", "-"),
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
    parser = argparse.ArgumentParser(description="Generate learning curve charts")
    parser.add_argument("--seeds", type=int, default=50, help="Number of random seeds (default: 50)")
    parser.add_argument("--config", type=str, default=None, help="Base MDP config (default: config/rule_based.yaml)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = args.config or str(repo_root / "config" / "rule_based.yaml")
    config = load_config(config_path)

    n_steps = config.episode_days * config.steps_per_day
    n_seeds = args.seeds
    window = min(20, n_steps)

    print(f"Config: {config_path}")
    print(f"MDP: {config.episode_days} days x {config.steps_per_day} steps = {n_steps} steps")
    print(f"Seeds: {n_seeds}\n")

    all_data: dict[str, np.ndarray] = {}
    for label, agent_dict, _color, _ls in AGENT_VARIANTS:
        agent_cfg = AgentConfig.model_validate(agent_dict)
        print(f"Running {label}...")
        all_data[label] = run_agent(config, agent_cfg, n_seeds)

    steps = np.arange(1, n_steps + 1)
    contextual_optimal = steps * (3 / 7)
    noncontextual_optimal = steps * 0.375

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for label, _agent_dict, color, ls in AGENT_VARIANTS:
        rewards = all_data[label]
        cum_mean = np.cumsum(rewards, axis=1).mean(axis=0)
        cum_std = np.cumsum(rewards, axis=1).std(axis=0)

        ax1.fill_between(steps, cum_mean - cum_std, cum_mean + cum_std, alpha=0.1, color=color)
        ax1.plot(steps, cum_mean, label=label, color=color, linewidth=1.5, linestyle=ls)

        step_mean = rewards.mean(axis=0)
        step_smooth = np.convolve(step_mean, np.ones(window) / window, mode="valid")
        window_steps = np.arange(window, n_steps + 1)
        ax2.plot(window_steps, step_smooth, label=label, color=color, linewidth=1.5, linestyle=ls)

    ax1.plot(steps, noncontextual_optimal, label="Non-ctx optimal", color="#2196F3", linewidth=1, linestyle="--", alpha=0.5)
    ax1.plot(steps, contextual_optimal, label="Ctx optimal", color="#4CAF50", linewidth=1, linestyle="--", alpha=0.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("Cumulative Reward")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=3 / 7, color="#4CAF50", linestyle="--", alpha=0.5, label="Ctx optimal")
    ax2.axhline(y=0.375, color="#2196F3", linestyle="--", alpha=0.5, label="Non-ctx optimal")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward per Step (rolling avg)")
    ax2.set_title(f"Per-Step Reward (window={window})")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.25, 0.50)

    plt.tight_layout()
    out = Path(__file__).parent / "images" / "learning_curves.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
