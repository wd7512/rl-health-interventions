"""Sweep epsilon values for the epsilon-greedy agent.

Usage:
    uv run python docs/experimental_phases/pearl_random/sweep_epsilon.py
"""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _shared import agent_label, resolve_config, run_agent_detailed

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import AgentConfig

_HIGH = 2

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
IMAGES_DIR = _HERE / "images"
CONFIG_PATH = resolve_config()

N_SEEDS = 50

ARM_COLORS: dict[str, str] = {
    "Control": "#4C72B0",
    "Random": "#DD8452",
    "Fixed COM-B": "#55A868",
    "RL (EG)": "#C44E52",
}


def _compute_burden_pct(trajectories: list[list[dict]]) -> np.ndarray:
    """Return per-step % medium+high burden averaged across seeds."""
    n_steps = len(trajectories[0])
    n_seeds = len(trajectories)
    burden_map = {"low": 0, "medium": 1, "high": 2}
    pcts = np.zeros(n_steps)
    for t in range(n_steps):
        non_low = sum(
            1
            for s in range(n_seeds)
            if burden_map.get(trajectories[s][t].get("burden", "low"), 0) > 0
        )
        pcts[t] = non_low / n_seeds * 100
    return pcts


def _baseline_label(cfg) -> str:
    label = agent_label(cfg)
    if label == "Fixed":
        if cfg.action == "idle":
            return "Control"
        return "Fixed"
    return label


def main() -> None:  # noqa: C901, PLR0915
    epsilons = [round(i * 0.1, 1) for i in range(11)]
    config = load_config(CONFIG_PATH)

    logger.info("Loading baseline arms...")
    baseline_rewards: dict[str, np.ndarray] = {}
    baseline_trajs: dict[str, list[list[dict]]] = {}
    for acfg in config.agents:
        label = _baseline_label(acfg)
        rewards, trajs = run_agent_detailed(config, acfg, N_SEEDS, agent_index=0)
        baseline_rewards[label] = rewards
        baseline_trajs[label] = trajs
        logger.info("  %s: total_reward=%.2f", label, np.sum(rewards, axis=1).mean())

    logger.info("Running epsilon sweep...")
    sweep_results: dict[float, np.ndarray] = {}
    sweep_trajs: dict[float, list[list[dict]]] = {}
    for eps in epsilons:
        agent_cfg = AgentConfig(type="epsilon_greedy", epsilon=eps)
        rewards, trajs = run_agent_detailed(config, agent_cfg, N_SEEDS, agent_index=0)
        sweep_results[eps] = rewards
        sweep_trajs[eps] = trajs
        logger.info("  ε=%.1f: total_reward=%.2f", eps, np.sum(rewards, axis=1).mean())

    logger.info("\n" + "=" * 180)
    header = (
        f"{'Arm':<16} {'Eps':<6} {'TotalRew':>10} {'SE':>8}"
        f" {'PkDay':>7} {'Pk%':>6} {'Sust%':>7}"
        f" {'Low%':>7} {'Med%':>7} {'Hi%':>7} {'Idle%':>7}"
    )
    logger.info(header)
    logger.info("-" * 180)

    burden_map = {"low": 0, "medium": 1, "high": 2}
    steps_per_day = config.steps_per_day
    episode_len = config.episode_days * steps_per_day
    sustained_start = 30 * steps_per_day
    sustained_end = episode_len

    rows: list[dict] = []

    # Baseline rows
    for arm_name in ["Control", "Random", "Fixed COM-B", "RL (EG)"]:
        if arm_name not in baseline_rewards:
            continue
        r = baseline_rewards[arm_name]
        t = baseline_trajs[arm_name]
        total_per_seed = np.sum(r, axis=1)
        mean_r = float(np.mean(total_per_seed))
        se_r = float(np.std(total_per_seed) / np.sqrt(N_SEEDS))
        pct = _compute_burden_pct(t)
        peak_idx = int(np.argmax(pct))
        peak_val = float(pct[peak_idx])
        sustained = (
            float(np.mean(pct[sustained_start:sustained_end]))
            if sustained_end > sustained_start
            else 0.0
        )

        all_burden = np.array(
            [
                [
                    burden_map.get(traj[step].get("burden", "low"), 0)
                    for step in range(episode_len)
                ]
                for traj in t
            ]
        )
        low_pct = np.mean(all_burden == 0) * 100
        med_pct = np.mean(all_burden == 1) * 100
        high_pct = np.mean(all_burden == _HIGH) * 100

        actions_flat = [
            traj[step]["action"] for traj in t for step in range(episode_len)
        ]
        idle_pct = sum(1 for a in actions_flat if a == "idle") / len(actions_flat) * 100

        eps_str = (
            "-"
            if arm_name == "RL (EG)"
            else ("-" if arm_name in ("Control", "Random", "Fixed COM-B") else "")
        )
        logger.info(
            f"{arm_name:<16} {eps_str:<6} {mean_r:>10.2f} {se_r:>8.3f} "
            f"{peak_idx:>7d} {peak_val:>6.1f} {sustained:>7.1f} "
            f"{low_pct:>7.1f} {med_pct:>7.1f} {high_pct:>7.1f} {idle_pct:>7.1f}"
        )
        rows.append(
            {
                "arm": arm_name,
                "eps": eps_str,
                "reward": mean_r,
                "se": se_r,
                "peak_day": peak_idx,
                "peak_pct": peak_val,
                "sustained": sustained,
                "low": low_pct,
                "med": med_pct,
                "high": high_pct,
                "idle": idle_pct,
            }
        )

    # Sweep rows
    for eps in epsilons:
        r = sweep_results[eps]
        t = sweep_trajs[eps]
        total_per_seed = np.sum(r, axis=1)
        mean_r = float(np.mean(total_per_seed))
        se_r = float(np.std(total_per_seed) / np.sqrt(N_SEEDS))
        pct = _compute_burden_pct(t)
        peak_idx = int(np.argmax(pct))
        peak_val = float(pct[peak_idx])
        sustained = (
            float(np.mean(pct[sustained_start:sustained_end]))
            if sustained_end > sustained_start
            else 0.0
        )

        all_burden = np.array(
            [
                [
                    burden_map.get(traj[step].get("burden", "low"), 0)
                    for step in range(episode_len)
                ]
                for traj in t
            ]
        )
        low_pct = np.mean(all_burden == 0) * 100
        med_pct = np.mean(all_burden == 1) * 100
        high_pct = np.mean(all_burden == _HIGH) * 100

        actions_flat = [
            traj[step]["action"] for traj in t for step in range(episode_len)
        ]
        idle_pct = sum(1 for a in actions_flat if a == "idle") / len(actions_flat) * 100

        logger.info(
            f"{'RL (EG)':<16} {eps:<6.1f} {mean_r:>10.2f} {se_r:>8.3f} "
            f"{peak_idx:>7d} {peak_val:>6.1f} {sustained:>7.1f} "
            f"{low_pct:>7.1f} {med_pct:>7.1f} {high_pct:>7.1f} {idle_pct:>7.1f}"
        )
        rows.append(
            {
                "arm": "RL (EG)",
                "eps": eps,
                "reward": mean_r,
                "se": se_r,
                "peak_day": peak_idx,
                "peak_pct": peak_val,
                "sustained": sustained,
                "low": low_pct,
                "med": med_pct,
                "high": high_pct,
                "idle": idle_pct,
            }
        )

    # Plot
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    sweep_only = [r for r in rows if isinstance(r["eps"], (int, float))]
    baselines = [r for r in rows if r["eps"] == "-"]

    epsilon_x = [r["eps"] for r in sweep_only]
    rewards_y = [r["reward"] for r in sweep_only]
    sustained_y = [r["sustained"] for r in sweep_only]

    color = "#C44E52"
    ax1.plot(
        epsilon_x,
        rewards_y,
        "o-",
        color=color,
        linewidth=2,
        markersize=6,
        label="Total Reward (RL EG)",
    )
    baseline_map = {b["arm"]: b for b in baselines}
    for arm_name, ls, label_fn in [
        ("Control", "--", lambda r: f"Control ({r:.1f})"),
        ("Random", ":", lambda r: f"Random ({r:.1f})"),
        ("Fixed COM-B", "-.", lambda r: f"Fixed COM-B ({r:.1f})"),
    ]:
        b = baseline_map.get(arm_name)
        if b is None:
            continue
        ax1.axhline(
            y=b["reward"],
            color=ARM_COLORS[arm_name],
            linestyle=ls,
            alpha=0.5,
            label=label_fn(b["reward"]),
        )
    ax1.set_xlabel("Epsilon (exploration rate)")
    ax1.set_ylabel("Total Reward (60 days)", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(epsilons)
    ax1.set_xticklabels([f"{e:.1f}" for e in epsilons])

    ax2 = ax1.twinx()
    ax2.plot(
        epsilon_x,
        sustained_y,
        "s--",
        color="#55A868",
        linewidth=2,
        markersize=6,
        label="Sustained Burden %",
    )
    ax2.set_ylabel("Burden (% medium+high, days 30-59)", color="#55A868")
    ax2.tick_params(axis="y", labelcolor="#55A868")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, frameon=True, fontsize=8, loc="lower left"
    )

    ax1.set_title(f"Epsilon Sweep: RL (EG) Agent Performance ({N_SEEDS} seeds)")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "epsilon_sweep.png", dpi=150)
    logger.info("Saved epsilon_sweep.png")
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
