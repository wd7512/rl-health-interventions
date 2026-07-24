"""Compare contextual (burden-aware) vs non-contextual epsilon-greedy agent.

Hypothesis:
  Contextual EG (context_features=["burden"]) will learn state-dependent
  Q-values: idle more when burden=high to let the window drain, engage
  when burden=low. This should reduce sustained burden modestly and
  increase total reward slightly vs non-contextual EG.

  But with random transition tables, the benefit will be small (3-8pp
  burden reduction) because no action is genuinely better in any state
  — strategic idling is the only lever.

Usage:
    uv run python docs/experimental_phases/pearl_random/bench_contextual.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from _shared import resolve_config, run_agent_detailed

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import AgentConfig

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
IMAGES_DIR = _HERE / "images"
CONFIG_PATH = resolve_config()

N_SEEDS = 50
_HIGH = 2

ARM_COLORS: dict[str, str] = {
    "Control": "#4C72B0",
    "Random": "#DD8452",
    "Fixed COM-B": "#55A868",
    "RL (EG)": "#C44E52",
    "RL (EG, ctx)": "#937860",
    "RL (EG, no-ctx)": "#C44E52",
}


def _compute_burden_pct(trajectories: list[list[dict]]) -> np.ndarray:
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


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    config = load_config(CONFIG_PATH)
    epoch_len = config.steps_per_day * config.episode_days
    sustained_start = 30 * config.steps_per_day
    burden_map = {"low": 0, "medium": 1, "high": 2}

    logger.info("Loading baseline arms...")
    records: dict[str, dict] = {}
    for acfg in config.agents:
        rewards, trajs = run_agent_detailed(config, acfg, N_SEEDS, agent_index=0)
        total = np.sum(rewards, axis=1)
        mean_r = float(np.mean(total))
        se_r = float(np.std(total) / np.sqrt(N_SEEDS))
        pct = _compute_burden_pct(trajs)
        peak_idx = int(np.argmax(pct))
        peak_val = float(pct[peak_idx])
        sustained = (
            float(np.mean(pct[sustained_start:]))
            if sustained_start < epoch_len
            else 0.0
        )
        all_burden = np.array(
            [
                [
                    burden_map.get(traj[step].get("burden", "low"), 0)
                    for step in range(epoch_len)
                ]
                for traj in trajs
            ]
        )
        low_pct = np.mean(all_burden == 0) * 100
        med_pct = np.mean(all_burden == 1) * 100
        high_pct = np.mean(all_burden == _HIGH) * 100
        actions_flat = [
            traj[step]["action"] for traj in trajs for step in range(epoch_len)
        ]
        idle_pct = sum(1 for a in actions_flat if a == "idle") / len(actions_flat) * 100

        label = (
            "Control"
            if (acfg.type == "fixed" and acfg.action == "idle")
            else acfg.type.replace("_", " ").title()
        )
        if label == "Random":
            label = "Random"
        elif label == "Comb Weighted Fixed":
            label = "Fixed COM-B"
        elif label == "Epsilon Greedy":
            label = "RL (EG) (config)"
        records[label] = {
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
        logger.info(
            "  %s: total=%.2f se=%.3f pk=%d (%.0f%%)"
            " sust=%.1f%% low=%.0f%% med=%.0f%% high=%.0f%% idle=%.0f%%",
            label, mean_r, se_r, peak_idx, peak_val,
            sustained, low_pct, med_pct, high_pct, idle_pct,
        )

    logger.info("Running EG variants (ε=0.3)...")
    for eps in [0.3]:
        for ctx_flag in [False, True]:
            kwargs: dict[str, Any] = {"type": "epsilon_greedy", "epsilon": eps}
            if ctx_flag:
                kwargs["contextual"] = True
                kwargs["context_features"] = ["burden"]
                label = "RL (EG) ctx-burden"
            else:
                label = "RL (EG) no-ctx"

            rewards, trajs = run_agent_detailed(
                config, AgentConfig(**kwargs), N_SEEDS, agent_index=0
            )
            total = np.sum(rewards, axis=1)
            mean_r = float(np.mean(total))
            se_r = float(np.std(total) / np.sqrt(N_SEEDS))
            pct = _compute_burden_pct(trajs)
            peak_idx = int(np.argmax(pct))
            peak_val = float(pct[peak_idx])
            sustained = (
                float(np.mean(pct[sustained_start:]))
                if sustained_start < epoch_len
                else 0.0
            )
            all_burden = np.array(
                [
                    [
                        burden_map.get(traj[step].get("burden", "low"), 0)
                        for step in range(epoch_len)
                    ]
                    for traj in trajs
                ]
            )
            low_pct = np.mean(all_burden == 0) * 100
            med_pct = np.mean(all_burden == 1) * 100
            high_pct = np.mean(all_burden == _HIGH) * 100
            actions_flat = [
                traj[step]["action"] for traj in trajs for step in range(epoch_len)
            ]
            idle_pct = (
                sum(1 for a in actions_flat if a == "idle") / len(actions_flat) * 100
            )
            records[label] = {
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
            logger.info(
                "  %s: total=%.2f se=%.3f pk=%d (%.0f%%)"
                " sust=%.1f%% low=%.0f%% med=%.0f%% high=%.0f%% idle=%.0f%%",
                label, mean_r, se_r, peak_idx, peak_val,
                sustained, low_pct, med_pct, high_pct, idle_pct,
            )

    # Table
    order = ["Control", "Random", "Fixed COM-B", "RL (EG) no-ctx", "RL (EG) ctx-burden"]
    logger.info("\n" + "=" * 160)
    hdr = (
        f"{'Arm':<22} {'TotalRew':>9} {'SE':>7}"
        f" {'PkDay':>6} {'Pk%':>5} {'Sust%':>6}"
        f" {'Low%':>6} {'Med%':>6} {'Hi%':>5} {'Idle%':>6}"
    )
    logger.info(hdr)
    logger.info("-" * 160)
    for name in order:
        if name not in records:
            continue
        r = records[name]
        logger.info(
            f"{name:<22} {r['reward']:>9.2f} {r['se']:>7.3f}"
            f" {r['peak_day']:>6d} {r['peak_pct']:>5.1f} {r['sustained']:>6.1f}"
            f" {r['low']:>6.1f} {r['med']:>6.1f} {r['high']:>5.1f} {r['idle']:>6.1f}"
        )

    # Plot: burden trajectory comparison
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: burden trajectory over time
    window = 5
    for label_key in ["RL (EG) no-ctx", "RL (EG) ctx-burden"]:
        if label_key not in records:
            continue
        # Re-run to get trajectories (or store from above — simplest is to re-run)
        kwargs: dict[str, Any] = {"type": "epsilon_greedy", "epsilon": 0.3}
        if "ctx" in label_key:
            kwargs["contextual"] = True
            kwargs["context_features"] = ["burden"]
        _, trajs = run_agent_detailed(
            config, AgentConfig(**kwargs), N_SEEDS, agent_index=0
        )
        pct = _compute_burden_pct(trajs)
        smoothed = np.convolve(pct, np.ones(window) / window, mode="valid")
        x_vals = np.arange(window // 2, epoch_len - window // 2)
        color = ARM_COLORS.get(label_key, "#C44E52")
        ax1.plot(
            x_vals,
            smoothed[: len(x_vals)],
            color=color,
            linewidth=1.8,
            label=label_key,
        )

    ax1.set_xlabel("Day")
    ax1.set_ylabel("Burden (% medium+high, MA-5)")
    ax1.set_title("Burden Trajectory: Contextual vs Non-contextual")
    ax1.legend(frameon=True, fontsize=8)

    # Right: bar chart
    groups = ["RL (EG) no-ctx", "RL (EG) ctx-burden"]
    metrics = ["reward", "sustained", "idle"]
    metric_labels = ["Total Reward", "Sustained Burden %", "Idle Rate %"]
    x = np.arange(len(metrics))
    w = 0.35
    for i, grp in enumerate(groups):
        if grp not in records:
            continue
        vals = [records[grp][m] for m in metrics]
        offset = (i - 0.5) * w
        ax2.bar(
            x + offset,
            vals,
            w,
            color=ARM_COLORS.get(grp, "#C44E52"),
            alpha=0.85,
            edgecolor="white",
            label=grp,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels)
    ax2.set_title("Key Metrics Comparison")
    ax2.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "contextual_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved contextual_comparison.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
