"""Generate PEARL-aligned visualizations from saved experiment data.

Usage:
    # 1. Run the full experiment with trajectory export:
    uv run python docs/experimental_phases/pearl_random/run_experiments.py \\
        --trajectories

    # 2. Generate plots:
    uv run python docs/experimental_phases/pearl_random/plots.py

Output: docs/experimental_phases/pearl_random/images/*.png
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
IMAGES_DIR = _HERE / "images"
FIXTURE_PATH = _HERE / "results" / "pearl_random.json"
TRAJECTORY_PATH = _HERE / "results" / "trajectories" / "pearl_random_trajectories.json"

FEATURE_IMPORTANCE = {
    "Pre-study steps mean": 1.23,
    "Weight": 1.19,
    "Age": 1.17,
    "Walk pattern": 1.13,
    "Steps StdDev": 1.02,
    "Recent steps mean": 0.88,
    "Recent walk pattern": 0.70,
    "Avg morning steps": 0.63,
}

THEMES = [
    "ability",
    "perceived_benefit",
    "planning",
    "prioritization",
    "social_opportunity",
    "physical_opportunity",
]

ARM_COLORS: dict[str, str] = {
    "Control": "#4C72B0",
    "Random": "#DD8452",
    "Fixed COM-B": "#55A868",
    "RL (EG)": "#C44E52",
}


def _arm_order() -> list[str]:
    return ["Control", "RL (EG)", "Random", "Fixed COM-B"]


def _theme_of(action: str) -> str:
    if action == "idle":
        return "idle"
    return action.rsplit("_", 1)[0]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )


def load_trajectories() -> dict[str, list[list[dict]]]:
    with open(TRAJECTORY_PATH) as f:
        data = json.load(f)
    trajectories: dict[str, list[list[dict]]] = {}
    for arm_label, seeds_data in data["arms"].items():
        seed_list: list[list[dict]] = []
        for sidx in range(1, data["n_seeds"] + 1):
            seed_list.append(seeds_data[f"seed_{sidx}"])
        trajectories[arm_label] = seed_list
    return trajectories, data["n_seeds"]


def load_fixture() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def fig2_total_reward(  # noqa: PLR0915
    trajectories: dict[str, list[list[dict]]], n_seeds: int
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), width_ratios=[1, 1.4])

    fixture = load_fixture()
    agents_data = fixture["agents"]
    labels = _arm_order()
    totals = [agents_data[lab]["total_reward"] for lab in labels]
    stds = [agents_data[lab]["total_std"] for lab in labels]
    n_fixture_seeds = fixture["seeds"]

    colors = [ARM_COLORS[lab] for lab in labels]
    ax1.bar(
        range(len(labels)),
        totals,
        yerr=[s / np.sqrt(n_fixture_seeds) for s in stds],
        capsize=4,
        color=colors,
        width=0.6,
        edgecolor="white",
    )
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Total Reward (60 days)")
    ax1.set_title(f"A: Cumulative Reward ({n_fixture_seeds} seeds)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    episode_len = len(trajectories[labels[0]][0])
    mid = episode_len // 2

    early_data: dict[str, float] = {}
    late_data: dict[str, float] = {}
    for label in labels:
        seed_rewards = trajectories[label]
        early_all = [sum(r["reward"] for r in seed[:mid]) for seed in seed_rewards]
        late_all = [sum(r["reward"] for r in seed[mid:]) for seed in seed_rewards]
        early_data[label] = np.mean(early_all)
        late_data[label] = np.mean(late_all)

    x = np.arange(len(labels))
    w = 0.35
    ax2.bar(
        x - w / 2,
        [early_data[lab] for lab in labels],
        w,
        label=f"Early (days 1\u2013{mid})",
        color=[ARM_COLORS[lab] for lab in labels],
        alpha=0.7,
        edgecolor="white",
    )
    ax2.bar(
        x + w / 2,
        [late_data[lab] for lab in labels],
        w,
        label=f"Late (days {mid + 1}\u2013{episode_len})",
        color=[ARM_COLORS[lab] for lab in labels],
        alpha=1.0,
        edgecolor="white",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Cumulative Reward")
    ax2.set_title(f"B: Early vs Late (avg {n_seeds} seeds)")
    ax2.legend(frameon=True, fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig2_total_reward.png", dpi=150)
    plt.close(fig)
    logger.info("Saved fig2_total_reward.png")


def fig3_daily_trajectory(
    trajectories: dict[str, list[list[dict]]], n_seeds: int
) -> None:
    labels = _arm_order()
    episode_len = len(trajectories[labels[0]][0])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    window = 5

    for label in labels:
        all_rewards = np.array(
            [[r["reward"] for r in seed] for seed in trajectories[label]]
        )
        mean_rewards = np.mean(all_rewards, axis=0)
        smoothed = np.convolve(mean_rewards, np.ones(window) / window, mode="valid")
        x_vals = np.arange(window // 2, episode_len - window // 2)
        ax.plot(
            x_vals,
            smoothed[: len(x_vals)],
            color=ARM_COLORS[label],
            label=label,
            linewidth=1.8,
        )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Day")
    ax.set_ylabel("Per-step Reward (MA-5)")
    ax.set_title(f"Daily Reward Trajectory by Arm (avg {n_seeds} seeds, 5-day MA)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig3_daily_trajectory.png", dpi=150)
    plt.close(fig)
    logger.info("Saved fig3_daily_trajectory.png")


def fig9_action_distribution(
    trajectories: dict[str, list[list[dict]]], n_seeds: int
) -> None:
    labels = _arm_order()
    themes_display = [
        "Idle",
        "Ability",
        "Benefit",
        "Planning",
        "Prioritization",
        "Social",
        "Physical",
    ]
    theme_keys = ["idle", *THEMES]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(themes_display))
    w = 0.2

    for i, label in enumerate(labels):
        theme_fracs = {t: [] for t in theme_keys}
        for seed in trajectories[label]:
            actions = [r["action"] for r in seed]
            total = len(actions)
            tc = Counter(_theme_of(a) for a in actions)
            for t in theme_keys:
                theme_fracs[t].append(tc.get(t, 0) / total * 100)
        means = [np.mean(theme_fracs[t]) for t in theme_keys]
        offset = (i - 1.5) * w
        ax.bar(
            x + offset,
            means,
            w,
            label=label,
            color=ARM_COLORS[label],
            alpha=0.85,
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(themes_display)
    ax.set_ylabel("Selection Frequency (%)")
    ax.set_title(f"Action Distribution by Arm (avg {n_seeds} seeds)")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig9_action_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved fig9_action_distribution.png")


def fig12_feature_importance() -> None:
    features = list(FEATURE_IMPORTANCE.keys())
    weights = list(FEATURE_IMPORTANCE.values())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors_positive = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
    bars = ax.barh(
        range(len(features)),
        weights,
        color=colors_positive,
        edgecolor="white",
        height=0.6,
    )
    ax.axvline(
        x=0.5,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label="Selection boundary (0.5)",
    )
    ax.legend(frameon=True, fontsize=9)

    for bar, w in zip(bars, weights, strict=True):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.2f}",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("XGBoost Feature Weight")
    ax.set_title("Fig 12: Most Influential Features (PEARL)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig12_feature_importance.png", dpi=150)
    plt.close(fig)
    logger.info("Saved fig12_feature_importance.png")


def burden_profile(trajectories: dict[str, list[list[dict]]], n_seeds: int) -> None:
    labels = _arm_order()
    episode_len = len(trajectories[labels[0]][0])
    burden_map = {"low": 0, "medium": 1, "high": 2}
    window = 5

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for label in labels:
        all_burden = np.array(
            [
                [burden_map.get(r.get("burden", "low"), 0) for r in seed]
                for seed in trajectories[label]
            ]
        )
        mean_burden = np.mean(all_burden, axis=0)
        smoothed = np.convolve(mean_burden, np.ones(window) / window, mode="valid")
        x_vals = np.arange(window // 2, episode_len - window // 2)
        ax.plot(
            x_vals,
            smoothed[: len(x_vals)],
            color=ARM_COLORS[label],
            label=label,
            linewidth=1.8,
        )

    ax.set_xlabel("Day")
    ax.set_ylabel("Burden Level (MA-5)")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_title(f"Burden Progression by Arm (avg {n_seeds} seeds, 5-day MA)")
    ax.legend(frameon=True)
    ax.set_ylim(-0.1, 2.1)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "burden_profile.png", dpi=150)
    plt.close(fig)
    logger.info("Saved burden_profile.png")


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAJECTORY_PATH.exists():
        logger.error(
            "Trajectory file not found at %s.\n"
            "Run the experiment first with trajectory export:\n"
            "  uv run python docs/experimental_phases/pearl_random/run_experiments.py "
            "--trajectories",
            TRAJECTORY_PATH,
        )
        return

    logger.info("Loading trajectory data from %s...", TRAJECTORY_PATH)
    trajectories, n_seeds = load_trajectories()
    logger.info("Loaded %d seeds for %d arms", n_seeds, len(trajectories))

    setup_style()

    fig2_total_reward(trajectories, n_seeds)
    fig3_daily_trajectory(trajectories, n_seeds)
    fig9_action_distribution(trajectories, n_seeds)
    fig12_feature_importance()
    burden_profile(trajectories, n_seeds)

    logger.info("All images saved to %s", IMAGES_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
