"""Generate learning curve and hyperparameter charts for MVP agents.

Loads agent configs, runs each variant, and produces cumulative/per-step
reward plots plus hyperparameter sensitivity charts (when results CSV exists).
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl_health_interventions.config.loader import load_config
from _shared import agent_label, resolve_config, run_agent
from optimal_bound import compute_bounds

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

IMAGES_DIR = Path(__file__).resolve().parent / "images"


def _read_csv_dict(path: Path) -> list[dict[str, str]]:
    """Read CSV file and return list of dictionaries."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_numeric(value: str) -> float | int | None:
    """Convert string to numeric (float or int). Returns None if not numeric."""
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return None


def _parse_eg_df(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    """Parse epsilon-greedy results into standard and contextual groups."""
    std = []
    ctx = []
    for row in rows:
        if row.get("agent") != "epsilon_greedy":
            continue
        params = row.get("params", "")
        if params.startswith("ctx_"):
            epsilon = _extract_float(params, r"eps=([\d.]+)")
            ctx.append({**row, "epsilon": epsilon})
        else:
            epsilon = _extract_float(params, r"eps=([\d.]+)")
            std.append({**row, "epsilon": epsilon})
    std.sort(key=lambda x: x["epsilon"])
    ctx.sort(key=lambda x: x["epsilon"])
    return std, ctx


def _extract_float(params: str, pattern: str) -> float:
    """Extract float value from params string using regex."""
    match = re.search(pattern, params)
    return float(match.group(1)) if match else 0.0


def _extract_int(params: str, pattern: str) -> int:
    """Extract int value from params string using regex."""
    match = re.search(pattern, params)
    return int(match.group(1)) if match else 0


def _parse_ucb_df(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    """Parse UCB results into standard and contextual groups."""
    std = []
    ctx = []
    for row in rows:
        if row.get("agent") != "ucb":
            continue
        params = row.get("params", "")
        if params.startswith("ctx_"):
            c_val = _extract_float(params, r"c=([\d.]+)")
            ctx.append({**row, "c_val": c_val})
        else:
            c_val = _extract_float(params, r"c=([\d.]+)")
            std.append({**row, "c_val": c_val})
    std.sort(key=lambda x: x["c_val"])
    ctx.sort(key=lambda x: x["c_val"])
    return std, ctx


def _parse_dec_df(rows: list[dict[str, str]], ctx: bool = False) -> list[dict]:
    """Parse decaying epsilon-greedy results."""
    result = []
    for row in rows:
        if row.get("agent") != "decaying_epsilon_greedy":
            continue
        params = row.get("params", "")
        is_ctx = params.startswith("ctx_")
        if ctx and not is_ctx:
            continue
        if not ctx and is_ctx:
            continue
        epsilon_start = _extract_float(params, r"eps_start=([\d.]+)")
        decay_steps = _extract_int(params, r"decay=(\d+)")
        result.append({**row, "epsilon_start": epsilon_start, "decay_steps": decay_steps})
    result.sort(key=lambda x: (x["epsilon_start"], x["decay_steps"]))
    return result


def _build_pivot(data: list[dict], row_key: str, col_key: str, val_key: str) -> dict:
    """Build pivot dictionary from list of dicts."""
    pivot: dict[float, dict[int, float]] = {}
    for row in data:
        row_val = row.get(row_key)
        col_val = row.get(col_key)
        val = row.get(val_key)
        if row_val is None or col_val is None or val is None:
            continue
        row_val = float(row_val)
        col_val = int(col_val)
        val = float(val)
        if row_val not in pivot:
            pivot[row_val] = {}
        pivot[row_val][col_val] = val
    return pivot


def _get_pivot_values(pivot: dict) -> tuple[list, list, list]:
    """Extract sorted indices and values from pivot dict."""
    row_keys = sorted(pivot.keys())
    col_keys = sorted(set(c for d in pivot.values() for c in d.keys()))
    values = [[pivot.get(r, {}).get(c, 0) for c in col_keys] for r in row_keys]
    return row_keys, col_keys, values


def generate_eg_chart(results_path: Path, suffix: str = "") -> None:
    rows = _read_csv_dict(results_path)
    std, ctx = _parse_eg_df(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        [r["epsilon"] for r in std],
        [float(r["total_mean"]) for r in std],
        yerr=[float(r["total_std"]) for r in std],
        fmt="-o",
        color="#FF9800",
        capsize=4,
        label="Standard EG",
    )
    ax.errorbar(
        [r["epsilon"] for r in ctx],
        [float(r["total_mean"]) for r in ctx],
        yerr=[float(r["total_std"]) for r in ctx],
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
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def generate_ucb_chart(results_path: Path, suffix: str = "") -> None:
    rows = _read_csv_dict(results_path)
    std, ctx = _parse_ucb_df(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        [r["c_val"] for r in std],
        [float(r["total_mean"]) for r in std],
        yerr=[float(r["total_std"]) for r in std],
        fmt="-o",
        color="#9C27B0",
        capsize=4,
        label="Standard UCB",
    )
    ax.errorbar(
        [r["c_val"] for r in ctx],
        [float(r["total_mean"]) for r in ctx],
        yerr=[float(r["total_std"]) for r in ctx],
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
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def generate_dec_heatmap(results_path: Path, suffix: str = "") -> None:
    rows = _read_csv_dict(results_path)
    dec = _parse_dec_df(rows, ctx=False)
    pivot = _build_pivot(dec, "epsilon_start", "decay_steps", "total_mean")
    row_keys, col_keys, values = _get_pivot_values(pivot)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(
        values,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Total Reward")

    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels(col_keys)
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels(row_keys)
    ax.set_xlabel("Decay Steps")
    ax.set_ylabel("Epsilon Start")
    ax.set_title("Standard D-EG: Total Reward by Epsilon Start and Decay Steps")

    max_val = max(max(row) for row in values) if values else 0
    for i in range(len(row_keys)):
        for j in range(len(col_keys)):
            val = values[i][j]
            ax.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center",
                color="white" if val < max_val * 0.7 else "black",
                fontsize=7,
            )

    plt.tight_layout()
    out = IMAGES_DIR / f"hyperparam_dec_heatmap{suffix}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate learning curve charts")
    parser.add_argument(
        "--seeds", type=int, default=50, help="Number of random seeds (default: 50)"
    )
    parser.add_argument("--config", type=str, default=None, help="Agent config filename")
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

    config_path = resolve_config(args.config)
    config = load_config(config_path)

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
    for agent_cfg in config.agents:
        label = agent_label(agent_cfg)
        logger.info("Running %s...", label)
        all_data[label] = run_agent(config, agent_cfg, n_seeds)

    steps = np.arange(1, n_steps + 1)
    try:
        bounds = compute_bounds(config)
    except ValueError as e:
        logger.warning("Could not compute optimal bounds: %s — skipping optimal lines", e)
        bounds = None
    if bounds is not None:
        ctx_per_step = bounds["contextual_optimal"]
        nctx_per_step = bounds["noncontextual_optimal"]
    else:
        ctx_per_step = nctx_per_step = 0.0
    contextual_optimal = steps * ctx_per_step * mask_frac
    noncontextual_optimal = steps * nctx_per_step * mask_frac

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, agent_cfg in enumerate(config.agents):
        label = agent_label(agent_cfg)
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

    if bounds is not None:
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

    if bounds is not None:
        ax2.axhline(
            y=ctx_per_step * mask_frac, color="#4CAF50", linestyle="--", alpha=0.5, label="Ctx optimal"
        )
        ax2.axhline(
            y=nctx_per_step * mask_frac, color="#2196F3", linestyle="--", alpha=0.5, label="Non-ctx optimal"
        )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward per Step (rolling avg)")
    ax2.set_title(f"Per-Step Reward (window={window})")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.25 * mask_frac, 0.50)

    plt.tight_layout()
    out = IMAGES_DIR / f"learning_curves{args.lc_suffix}.pdf"
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
