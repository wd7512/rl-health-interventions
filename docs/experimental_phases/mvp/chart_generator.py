"""Generate learning curve and hyperparameter charts for MVP agents.

Loads agent configs from mvp_extensions.yaml, runs each variant,
and produces cumulative/per-step reward plots plus hyperparameter
sensitivity charts (when results CSV exists).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_health_interventions.config.loader import load_config
from rl_health_interventions.experiment import run_episode
from rl_health_interventions.agents import derive_agent_seed, make as make_agent

logger = logging.getLogger(__name__)

_COLORS = [
    "#2196F3",
    "#4CAF50",
    "#FF9800",
    "#E91E63",
    "#9C27B0",
    "#00BCD4",
    "#795548",
    "#607D8B",
    "#F44336",
]
_LINESTYLES = ["-", "--", "-.", ":"]
_SHORT_NAMES: dict[str, str] = {
    "thompson_sampling": "TS",
    "epsilon_greedy": "EG",
    "ucb": "UCB",
    "decaying_epsilon_greedy": "D-EG",
    "random": "Random",
}


def _agent_label(cfg) -> str:
    if cfg.type == "random":
        return "Random"
    short = _SHORT_NAMES.get(cfg.type, cfg.type)
    prefix = "Ctx" if cfg.contextual else "Std"
    return f"{prefix} {short}"


def run_agent(config, agent_cfg, n_seeds: int) -> np.ndarray:
    """Run one agent variant over n_seeds. Returns per-step rewards (n_seeds, n_steps)."""
    exclude = {"type"}
    if not agent_cfg.contextual:
        exclude |= {"contextual", "context_feature"}
    rewards = []
    for seed in range(1, n_seeds + 1):
        kwargs = agent_cfg.model_dump(exclude=exclude, exclude_none=True)
        kwargs["actions"] = config.actions
        kwargs["seed"] = derive_agent_seed(seed, agent_index=0)
        agent = make_agent(agent_cfg.type, **kwargs)
        df = run_episode(config, agent, seed=seed)
        rewards.append(df["reward"].values)
    return np.array(rewards)


IMAGES_DIR = Path(__file__).parent / "images"


def _parse_eg_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eg = df[df["agent"] == "epsilon_greedy"].copy()
    eg["epsilon"] = eg["params"].str.extract(r"eps=([\d.]+)").astype(float)
    std = eg[~eg["params"].str.startswith("ctx_")].sort_values("epsilon")
    ctx = eg[eg["params"].str.startswith("ctx_")].sort_values("epsilon")
    return std, ctx


def _parse_ucb_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ucb = df[df["agent"] == "ucb"].copy()
    ucb["c_val"] = ucb["params"].str.extract(r"c=([\d.]+)").astype(float)
    std = ucb[~ucb["params"].str.startswith("ctx_")].sort_values("c_val")
    ctx = ucb[ucb["params"].str.startswith("ctx_")].sort_values("c_val")
    return std, ctx


def _parse_dec_df(df: pd.DataFrame, ctx: bool = False) -> pd.DataFrame:
    dec = df[df["agent"] == "decaying_epsilon_greedy"].copy()
    if ctx:
        dec = dec[dec["params"].str.startswith("ctx_")]
    else:
        dec = dec[~dec["params"].str.startswith("ctx_")]
    dec["epsilon_start"] = (
        dec["params"].str.extract(r"eps_start=([\d.]+)").astype(float)
    )
    dec["decay_steps"] = dec["params"].str.extract(r"decay=(\d+)").astype(int)
    return dec.sort_values(["epsilon_start", "decay_steps"])


def generate_eg_chart(results_path: Path, suffix: str = "") -> None:
    df = pd.read_csv(results_path)
    std, ctx = _parse_eg_df(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        std["epsilon"],
        std["total_mean"],
        yerr=std["total_std"],
        fmt="-o",
        color="#FF9800",
        capsize=4,
        label="Standard EG",
    )
    ax.errorbar(
        ctx["epsilon"],
        ctx["total_mean"],
        yerr=ctx["total_std"],
        fmt="-s",
        color="#E91E63",
        capsize=4,
        label="Contextual EG",
    )
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Total Reward")
    ax.set_title("Epsilon-Greedy: Epsilon vs Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = IMAGES_DIR / f"hyperparam_eg{suffix}.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def generate_ucb_chart(results_path: Path, suffix: str = "") -> None:
    df = pd.read_csv(results_path)
    std, ctx = _parse_ucb_df(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        std["c_val"],
        std["total_mean"],
        yerr=std["total_std"],
        fmt="-o",
        color="#9C27B0",
        capsize=4,
        label="Standard UCB",
    )
    ax.errorbar(
        ctx["c_val"],
        ctx["total_mean"],
        yerr=ctx["total_std"],
        fmt="-s",
        color="#00BCD4",
        capsize=4,
        label="Contextual UCB",
    )
    ax.set_xlabel("c (exploration constant)")
    ax.set_ylabel("Total Reward")
    ax.set_title("UCB: c vs Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = IMAGES_DIR / f"hyperparam_ucb{suffix}.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def generate_dec_heatmap(results_path: Path, suffix: str = "") -> None:
    df = pd.read_csv(results_path)
    dec = _parse_dec_df(df, ctx=False)

    pivot = dec.pivot_table(
        index="epsilon_start",
        columns="decay_steps",
        values="total_mean",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Total Reward")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Decay Steps")
    ax.set_ylabel("Epsilon Start")
    ax.set_title("Standard D-EG: Total Reward by Epsilon Start and Decay Steps")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center",
                color="white" if val < pivot.values.max() * 0.7 else "black",
                fontsize=7,
            )

    plt.tight_layout()
    out = IMAGES_DIR / f"hyperparam_dec_heatmap{suffix}.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate learning curve charts")
    parser.add_argument(
        "--seeds", type=int, default=50, help="Number of random seeds (default: 50)"
    )
    parser.add_argument("--config", type=str, default=None, help="Agent config file")
    parser.add_argument(
        "--hp-results",
        type=str,
        default="hyperparam_results.csv",
        help="Hyperparameter results CSV filename",
    )
    parser.add_argument(
        "--hp-suffix",
        type=str,
        default="",
        help="Suffix for hyperparameter chart filenames (e.g. '_masked')",
    )
    parser.add_argument(
        "--lc-suffix",
        type=str,
        default="",
        help="Suffix for learning curve filenames (e.g. '_masked')",
    )
    args = parser.parse_args()

    config_path = args.config or str(
        Path(__file__).parent / "configs" / "mvp_extensions.yaml"
    )
    raw = Path(config_path)
    if not raw.is_absolute():
        raw = Path(__file__).parent / "configs" / raw.name
    config = load_config(str(raw))

    multiplier = getattr(config, "reward_multiplier_by_step", None)
    mask_frac = float(np.mean(multiplier)) if multiplier else 1.0

    n_steps = config.episode_days * config.steps_per_day
    n_seeds = args.seeds
    window = min(20, n_steps)

    logger.info("Config: %s", config_path)
    logger.info(
        "MDP: %d days x %d steps = %d steps",
        config.episode_days,
        config.steps_per_day,
        n_steps,
    )
    logger.info("Seeds: %d\n", n_seeds)

    all_data: dict[str, np.ndarray] = {}
    for i, agent_cfg in enumerate(config.agents):
        label = _agent_label(agent_cfg)
        logger.info("Running %s...", label)
        all_data[label] = run_agent(config, agent_cfg, n_seeds)

    steps = np.arange(1, n_steps + 1)
    contextual_optimal = steps * (3 / 7) * mask_frac
    noncontextual_optimal = steps * 0.375 * mask_frac

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, agent_cfg in enumerate(config.agents):
        label = _agent_label(agent_cfg)
        rewards = all_data[label]
        color = _COLORS[i % len(_COLORS)]
        ls = _LINESTYLES[i % len(_LINESTYLES)]

        cum_mean = np.cumsum(rewards, axis=1).mean(axis=0)
        cum_std = np.cumsum(rewards, axis=1).std(axis=0)

        ax1.fill_between(
            steps, cum_mean - cum_std, cum_mean + cum_std, alpha=0.1, color=color
        )
        ax1.plot(steps, cum_mean, label=label, color=color, linewidth=1.5, linestyle=ls)

        step_mean = rewards.mean(axis=0)
        step_smooth = np.convolve(step_mean, np.ones(window) / window, mode="valid")
        window_steps = np.arange(window, n_steps + 1)
        ax2.plot(
            window_steps,
            step_smooth,
            label=label,
            color=color,
            linewidth=1.5,
            linestyle=ls,
        )

    ax1.plot(
        steps,
        noncontextual_optimal,
        label="Non-ctx optimal",
        color="#2196F3",
        linewidth=1,
        linestyle="--",
        alpha=0.5,
    )
    ax1.plot(
        steps,
        contextual_optimal,
        label="Ctx optimal",
        color="#4CAF50",
        linewidth=1,
        linestyle="--",
        alpha=0.5,
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("Cumulative Reward")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.axhline(
        y=3 / 7 * mask_frac, color="#4CAF50", linestyle="--", alpha=0.5, label="Ctx optimal"
    )
    ax2.axhline(
        y=0.375 * mask_frac, color="#2196F3", linestyle="--", alpha=0.5, label="Non-ctx optimal"
    )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward per Step (rolling avg)")
    ax2.set_title(f"Per-Step Reward (window={window})")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.25 * mask_frac, 0.50)

    plt.tight_layout()
    out = Path(__file__).parent / "images" / f"learning_curves{args.lc_suffix}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("\nSaved to %s", out)

    results_path = Path(__file__).parent / args.hp_results
    if results_path.exists():
        logger.info("\n--- Generating hyperparameter charts ---")
        generate_eg_chart(results_path, args.hp_suffix)
        generate_ucb_chart(results_path, args.hp_suffix)
        generate_dec_heatmap(results_path, args.hp_suffix)
    else:
        logger.warning("Skipping hyperparameter charts: %s not found", results_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
