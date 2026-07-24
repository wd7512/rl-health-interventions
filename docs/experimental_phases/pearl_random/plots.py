"""Generate PEARL-aligned visualizations from experiment data.

Usage:
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

from rl_health_interventions.agents import derive_agent_seed
from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.episode import run_episode

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
IMAGES_DIR = _HERE / "images"
FIXTURE_PATH = _HERE / "results" / "pearl_random.json"

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

THEME_DISPLAY = {
    "idle": "Idle",
    "ability": "Ability",
    "perceived_benefit": "Benefit",
    "planning": "Planning",
    "prioritization": "Prioritization",
    "social_opportunity": "Social",
    "physical_opportunity": "Physical",
}

ARM_COLORS: dict[str, str] = {
    "Control": "#4C72B0",
    "Random": "#DD8452",
    "Fixed COM-B": "#55A868",
    "RL (EG)": "#C44E52",
}

_SHORT_NAMES: dict[str, str] = {
    "epsilon_greedy": "RL (EG)",
    "comb_weighted_fixed": "Fixed COM-B",
    "fixed": "Fixed",
    "random": "Random",
}


def agent_label(cfg) -> str:
    if cfg.type == "fixed" and cfg.action == "idle":
        return "Control"
    return _SHORT_NAMES.get(cfg.type, cfg.type)


def _arm_order() -> list[str]:
    return ["Control", "RL (EG)", "Random", "Fixed COM-B"]


def _theme_of(action: str) -> str:
    if action == "idle":
        return "idle"
    return action.rsplit("_", 1)[0]


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })


def collect_trajectories(
    config_path: str, n_seeds: int = 1
) -> dict[str, list[list[dict]]]:
    config = load_config(config_path)
    trajectories: dict[str, list[list[dict]]] = {}
    for i, agent_cfg in enumerate(config.agents):
        label = agent_label(agent_cfg)
        exclude = {"type"}
        if not agent_cfg.contextual:
            exclude |= {"contextual", "context_features"}
        base_kwargs = agent_cfg.model_dump(exclude=exclude, exclude_none=True)
        base_kwargs["actions"] = config.action_names
        seed_records: list[list[dict]] = []
        for seed in range(1, n_seeds + 1):
            kwargs = base_kwargs.copy()
            kwargs["seed"] = derive_agent_seed(seed, agent_index=i)
            agent = make_agent(agent_cfg.type, **kwargs)
            records = run_episode(config, agent, seed=seed)
            seed_records.append(records)
        trajectories[label] = seed_records
    return trajectories


def load_fixture() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def fig2_total_reward(trajectories: dict) -> None:  # noqa: PLR0915
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 4.5), width_ratios=[1, 1.4]
    )

    fixture = load_fixture()
    agents_data = fixture["agents"]
    labels = _arm_order()
    totals = [agents_data[lab]["total_reward"] for lab in labels]
    stds = [agents_data[lab]["total_std"] for lab in labels]
    n_seeds = fixture["seeds"]

    colors = [ARM_COLORS[lab] for lab in labels]
    ax1.bar(
        range(len(labels)),
        totals,
        yerr=[s / np.sqrt(n_seeds) for s in stds],
        capsize=4,
        color=colors,
        width=0.6,
        edgecolor="white",
    )
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Total Reward (60 days)")
    ax1.set_title("A: Cumulative Reward (50 seeds)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    records = trajectories[next(iter(trajectories))][0]
    episode_len = len(records)
    mid = episode_len // 2

    early_data: dict[str, float] = {}
    late_data: dict[str, float] = {}
    for label in labels:
        r = trajectories[label][0]
        early = sum(step["reward"] for step in r[:mid])
        late = sum(step["reward"] for step in r[mid:])
        early_data[label] = early
        late_data[label] = late

    x = np.arange(len(labels))
    w = 0.35
    ax2.bar(
        x - w / 2,
        [early_data[lab] for lab in labels],
        w,
        label="Early (days 1\u201330)",
        color=[ARM_COLORS[lab] for lab in labels],
        alpha=0.7,
        edgecolor="white",
    )
    ax2.bar(
        x + w / 2,
        [late_data[lab] for lab in labels],
        w,
        label="Late (days 31\u201360)",
        color=[ARM_COLORS[lab] for lab in labels],
        alpha=1.0,
        edgecolor="white",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Cumulative Reward")
    ax2.set_title("B: Early vs Late (1 seed)")
    ax2.legend(frameon=True, fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig2_total_reward.png", dpi=150)
    plt.close(fig)
    logger.info("Saved fig2_total_reward.png")


def fig3_daily_trajectory(trajectories: dict) -> None:
    labels = _arm_order()
    records = trajectories[labels[0]][0]
    episode_len = len(records)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    window = 5

    for label in labels:
        r = trajectories[label][0]
        rewards = np.array([step["reward"] for step in r])
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        x_vals = np.arange(window // 2, episode_len - window // 2)
        ax.plot(
            x_vals, smoothed[:len(x_vals)],
            color=ARM_COLORS[label],
            label=label,
            linewidth=1.8,
        )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Day")
    ax.set_ylabel("Per-step Reward (MA-5)")
    ax.set_title(
        "Daily Reward Trajectory by Arm (1 seed, 5-day moving avg)"
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig3_daily_trajectory.png", dpi=150)
    plt.close(fig)
    logger.info("Saved fig3_daily_trajectory.png")


def fig9_action_distribution(trajectories: dict) -> None:
    labels = _arm_order()
    themes_display = ["Idle", "Ability", "Benefit", "Planning",
                      "Prioritization", "Social", "Physical"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(themes_display))
    w = 0.2

    for i, label in enumerate(labels):
        r = trajectories[label][0]
        actions = [step["action"] for step in r]
        theme_counts: Counter = Counter()
        for a in actions:
            theme_counts[_theme_of(a)] += 1
        total = len(actions)
        counts = []
        for t in ["idle", *THEMES]:
            c = theme_counts.get(t, 0)
            counts.append(c / total * 100)
        offset = (i - 1.5) * w
        ax.bar(
            x + offset, counts, w, label=label,
            color=ARM_COLORS[label], alpha=0.85, edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(themes_display)
    ax.set_ylabel("Selection Frequency (%)")
    ax.set_title("Action Distribution by Arm (1 seed)")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig9_action_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved fig9_action_distribution.png")


def fig12_feature_importance() -> None:
    features = list(FEATURE_IMPORTANCE.keys())
    weights = list(FEATURE_IMPORTANCE.values())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors_positive = plt.cm.Blues(
        np.linspace(0.4, 0.9, len(features))
    )
    bars = ax.barh(
        range(len(features)), weights, color=colors_positive,
        edgecolor="white", height=0.6,
    )
    ax.axvline(
        x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
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


def burden_profile(trajectories: dict) -> None:
    labels = _arm_order()
    records = trajectories[labels[0]][0]
    episode_len = len(records)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    burden_map = {"low": 0, "medium": 1, "high": 2}
    window = 5

    for label in labels:
        r = trajectories[label][0]
        burden_series = np.array([
            burden_map.get(step.get("burden", "low"), 0) for step in r
        ])
        smoothed = np.convolve(
            burden_series, np.ones(5) / 5, mode="valid"
        )
        x_vals = np.arange(window // 2, episode_len - window // 2)
        ax.plot(
            x_vals, smoothed[:len(x_vals)],
            color=ARM_COLORS[label], label=label, linewidth=1.8,
        )

    ax.set_xlabel("Day")
    ax.set_ylabel("Burden Level (MA-5)")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_title(
        "Burden Progression by Arm (1 seed, 5-day moving avg)"
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "burden_profile.png", dpi=150)
    plt.close(fig)
    logger.info("Saved burden_profile.png")


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config_path = str(_HERE / "configs" / "pearl_random.yaml")

    logger.info("Running single-seed trajectories for each arm...")
    trajectories = collect_trajectories(config_path, n_seeds=1)

    setup_style()

    fig2_total_reward(trajectories)
    fig3_daily_trajectory(trajectories)
    fig9_action_distribution(trajectories)
    fig12_feature_importance()
    burden_profile(trajectories)

    logger.info("All images saved to %s", IMAGES_DIR)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s"
    )
    main()
