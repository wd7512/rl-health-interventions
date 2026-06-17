"""Generate learning curve chart for contextual Thompson Sampling.

Excluded from ty/ruff via pyproject.toml tool settings.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.experiment import run_episode
from rl_health_interventions.agents import derive_agent_seed, make as make_agent


def run_seeds(config_path: str, n_seeds: int = 50) -> np.ndarray:
    """Run agent over n_seeds, return (n_seeds, 450) cumulative reward array."""
    all_rewards = []
    for seed in range(1, n_seeds + 1):
        config = load_config(config_path)
        agent_cfg = config.agents[0]
        kwargs = {
            k: v
            for k, v in agent_cfg.model_dump().items()
            if v is not None and k != "type"
        }
        kwargs["actions"] = config.actions
        kwargs["seed"] = derive_agent_seed(config.seed, agent_index=0)
        agent = make_agent(agent_cfg.type, **kwargs)
        df = run_episode(config, agent, seed=seed)
        all_rewards.append(df["reward"].values)
    return np.array(all_rewards)


def main() -> None:
    n_seeds = 50
    print(f"Running {n_seeds} seeds for each agent...")

    baseline = run_seeds("config/rule_based.yaml", n_seeds)
    contextual = run_seeds("config/contextual_thompson.yaml", n_seeds)

    # Cumulative reward per step (mean across seeds)
    baseline_cum = np.cumsum(baseline, axis=1).mean(axis=0)
    contextual_cum = np.cumsum(contextual, axis=1).mean(axis=0)

    # Per-step rolling average (window=20) for smoother curves
    window = 20
    baseline_step = np.convolve(
        baseline.mean(axis=0), np.ones(window) / window, mode="valid"
    )
    contextual_step = np.convolve(
        contextual.mean(axis=0), np.ones(window) / window, mode="valid"
    )

    # Theoretical upper bounds (constant per-step rates)
    steps = np.arange(1, 451)
    contextual_optimal = steps * (3 / 7)  # 0.4286 per step
    noncontextual_optimal = steps * 0.375  # always nudge

    # Confidence bands (std across seeds)
    baseline_cum_std = np.cumsum(baseline, axis=1).std(axis=0)
    contextual_cum_std = np.cumsum(contextual, axis=1).std(axis=0)

    # --- Figure 1: Cumulative reward over time ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.fill_between(
        steps,
        baseline_cum - baseline_cum_std,
        baseline_cum + baseline_cum_std,
        alpha=0.15,
        color="#2196F3",
    )
    ax1.fill_between(
        steps,
        contextual_cum - contextual_cum_std,
        contextual_cum + contextual_cum_std,
        alpha=0.15,
        color="#4CAF50",
    )
    ax1.plot(steps, baseline_cum, label="Standard TS", color="#2196F3", linewidth=1.5)
    ax1.plot(
        steps, contextual_cum, label="Contextual TS", color="#4CAF50", linewidth=1.5
    )
    ax1.plot(
        steps,
        noncontextual_optimal,
        label="Non-contextual optimal",
        color="#2196F3",
        linewidth=1,
        linestyle="--",
        alpha=0.6,
    )
    ax1.plot(
        steps,
        contextual_optimal,
        label="Contextual optimal",
        color="#4CAF50",
        linewidth=1,
        linestyle="--",
        alpha=0.6,
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("Cumulative Reward")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Figure 2: Per-step reward (rolling average) ---
    window_steps = np.arange(window, 451)
    ax2.plot(window_steps, baseline_step, label="Standard TS", color="#2196F3", linewidth=1.5)
    ax2.plot(
        window_steps, contextual_step, label="Contextual TS", color="#4CAF50", linewidth=1.5
    )
    ax2.axhline(y=3 / 7, color="#4CAF50", linestyle="--", alpha=0.6, label="Contextual optimal")
    ax2.axhline(y=0.375, color="#2196F3", linestyle="--", alpha=0.6, label="Non-contextual optimal")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward per Step (rolling avg)")
    ax2.set_title(f"Per-Step Reward (window={window})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.25, 0.50)

    plt.tight_layout()
    out = Path(__file__).parent / "images" / "learning_curves.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")

    # Print summary
    print(f"\nFinal cumulative reward (step 450):")
    print(f"  Standard TS:    {baseline_cum[-1]:.1f} +/- {baseline_cum_std[-1]:.1f}")
    print(f"  Contextual TS:  {contextual_cum[-1]:.1f} +/- {contextual_cum_std[-1]:.1f}")
    print(f"  Contextual opt: {contextual_optimal[-1]:.1f}")
    print(f"\nPer-step reward (last 50 steps average):")
    print(f"  Standard TS:    {baseline.mean(axis=0)[-50:].mean():.4f}")
    print(f"  Contextual TS:  {contextual.mean(axis=0)[-50:].mean():.4f}")
    print(f"  Contextual opt: {3/7:.4f}")


if __name__ == "__main__":
    main()
