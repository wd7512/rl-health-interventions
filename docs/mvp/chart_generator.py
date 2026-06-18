"""Generate learning curve charts for all contextual bandit agents.

Excluded from ty/ruff via pyproject.toml tool settings.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.experiment import run_episode
from rl_health_interventions.agents import derive_agent_seed, make as make_agent

# Agent definitions: (label, config_type, kwargs, color, linestyle)
AGENTS = [
    ("Standard TS", "thompson_sampling", {"alpha_prior": 1, "beta_prior": 1}, "#2196F3", "-"),
    ("Contextual TS", "thompson_sampling", {"alpha_prior": 1, "beta_prior": 1, "contextual": True, "context_feature": "activity"}, "#4CAF50", "-"),
    ("Standard EG", "epsilon_greedy", {"epsilon": 0.1}, "#FF9800", "-"),
    ("Contextual EG", "epsilon_greedy", {"epsilon": 0.1, "contextual": True, "context_feature": "activity"}, "#E91E63", "-"),
    ("Standard UCB", "ucb", {"c": 2.0}, "#9C27B0", "-"),
    ("Contextual UCB", "ucb", {"c": 2.0, "contextual": True, "context_feature": "activity"}, "#00BCD4", "-"),
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
    n_seeds = 50
    window = 20
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(str(repo_root / "config" / "contextual_thompson.yaml"))

    all_data: dict[str, np.ndarray] = {}
    for label, agent_type, kwargs, _color, _ls in AGENTS:
        print(f"Running {label} ({n_seeds} seeds)...")
        all_data[label] = run_seeds(agent_type, kwargs, config, n_seeds)

    n_steps = all_data[list(all_data.keys())[0]].shape[1]
    steps = np.arange(1, n_steps + 1)

    # Theoretical bounds
    contextual_optimal = steps * (3 / 7)
    noncontextual_optimal = steps * 0.375

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for label, _type, _kwargs, color, ls in AGENTS:
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
    print(f"Saved to {out}")

    # Print summary
    print(f"\nFinal cumulative reward (step {n_steps}):")
    for label in all_data:
        r = all_data[label]
        cum = np.cumsum(r, axis=1).mean(axis=0)
        cum_std = np.cumsum(r, axis=1).std(axis=0)
        print(f"  {label:<20} {cum[-1]:>6.1f} +/- {cum_std[-1]:.1f}")
    print(f"  {'Ctx optimal':<20} {contextual_optimal[-1]:>6.1f}")
    print(f"  {'Non-ctx optimal':<20} {noncontextual_optimal[-1]:>6.1f}")

    print(f"\nPer-step reward (last {window} steps average):")
    for label in all_data:
        r = all_data[label]
        print(f"  {label:<20} {r[:, -window:].mean():.4f}")
    print(f"  {'Ctx optimal':<20} {3/7:.4f}")


if __name__ == "__main__":
    main()
