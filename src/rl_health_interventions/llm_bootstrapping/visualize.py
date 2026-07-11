"""Black-and-white transition matrix charts with complete directed edges.

Writes matrix figures to ``docs/figures/<model>/matrices``.

Usage:
    uv run python -m rl_health_interventions.llm_bootstrapping.visualize \\
        --files data/bootstrap/results_deepseek.jsonl \\
               data/bootstrap/results_glm5.2.jsonl \\
        --fig-dir docs/figures
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as PlotPath

from rl_health_interventions.llm_bootstrapping._analysis import (
    SAMPLES_DAY_BOUNDARY,
    SAMPLES_WITHIN_DAY,
    TIMESTEPS,
    WITHIN_DAY_CELLS_PER_TIMESTEP,
    WITHIN_DAY_START,
    day_boundary_params,
    load,
    model_label,
    parse_record,
    within_day_params,
)
from rl_health_interventions.llm_bootstrapping.prompts import (
    ACTIONS,
    BURDENS,
    DAY_TYPES,
    SLEEP_TYPES,
    STEP_BINS,
)

logger = logging.getLogger(__name__)

NODE_RADIUS = 0.1
STEP_NODE_ABBR = {"inactive": "i", "moderate": "m", "active": "a"}

_POS = {
    "inactive": np.array([-0.45, -0.26]),
    "moderate": np.array([0.0, 0.52]),
    "active": np.array([0.45, -0.26]),
}
_SLEEP_POS = {"good": np.array([-0.4, 0.0]), "poor": np.array([0.4, 0.0])}


def aggregate_day_boundary(records: list[dict]) -> dict:
    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[tuple, int] = defaultdict(int)
    for rec in records:
        if rec["index"] >= WITHIN_DAY_START:
            continue
        cat, parsed = parse_record(rec)
        if cat != "day_boundary" or not isinstance(parsed, str):
            continue
        cell_idx = rec["index"] // SAMPLES_DAY_BOUNDARY
        params = day_boundary_params(cell_idx)
        key = (
            params["step_bin_daily"],
            params["burden"],
            params["day_type"],
            params["sleep"],
        )
        counts[key][parsed] += 1
        totals[key] += 1
    return {
        key: {o: c / totals[key] for o, c in outcomes.items()}
        for key, outcomes in counts.items()
    }


def aggregate_within_day(records: list[dict]) -> dict[int, dict]:
    all_probs: dict[int, dict] = {}
    for timestep_idx in range(TIMESTEPS):
        counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        totals: dict[tuple, int] = defaultdict(int)
        for rec in records:
            _count_within_day_record(rec, timestep_idx, counts, totals)
        all_probs[timestep_idx] = {
            key: {o: c / totals[key] for o, c in outcomes.items()}
            for key, outcomes in counts.items()
        }
    return all_probs


def _count_within_day_record(
    rec: dict,
    timestep_idx: int,
    counts: dict[tuple, dict[str, int]],
    totals: dict[tuple, int],
) -> None:
    if rec["index"] < WITHIN_DAY_START:
        return
    offset = rec["index"] - WITHIN_DAY_START
    samples_per_timestep = WITHIN_DAY_CELLS_PER_TIMESTEP * SAMPLES_WITHIN_DAY
    if offset // samples_per_timestep != timestep_idx:
        return

    within_off = offset % samples_per_timestep
    cell_idx = within_off // SAMPLES_WITHIN_DAY
    params = within_day_params(cell_idx)
    cat, parsed = parse_record(rec)
    if cat != "within_day" or not isinstance(parsed, tuple):
        return

    _, next_step_bin = parsed
    if not isinstance(next_step_bin, str):
        return
    key = (
        params["step_bin"],
        params["burden"],
        params["action"],
        params["day_type"],
        params["sleep"],
    )
    counts[key][next_step_bin] += 1
    totals[key] += 1


def aggregate_within_day_averaged(records: list[dict]) -> dict:
    per_t = aggregate_within_day(records)
    all_keys = set().union(*(t_probs.keys() for t_probs in per_t.values()))

    averaged: dict = {}
    for key in all_keys:
        per_outcome: dict[str, list[float]] = defaultdict(list)
        available_timesteps = 0
        for timestep_idx in range(TIMESTEPS):
            if key not in per_t.get(timestep_idx, {}):
                continue
            available_timesteps += 1
            for outcome, prob in per_t[timestep_idx][key].items():
                per_outcome[outcome].append(prob)
        if available_timesteps:
            averaged[key] = {
                outcome: sum(probs) / available_timesteps
                for outcome, probs in per_outcome.items()
            }
    return averaged


def _save_tables(db_probs: dict, wd_probs: dict[int, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {
        "|".join(str(value) for value in key): {"_": outcomes}
        for key, outcomes in db_probs.items()
    }
    with (out_dir / "day_boundary.json").open("w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2)

    for timestep_idx in range(TIMESTEPS):
        table = {
            "|".join(str(value) for value in key): {"_": outcomes}
            for key, outcomes in wd_probs.get(timestep_idx, {}).items()
        }
        with (out_dir / f"within_day_{timestep_idx}.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(table, fh, indent=2)
    logger.info("  Wrote 6 tables to %s/", out_dir)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([0.0, 1.0])
    return vec / norm


def _rotate(vec: np.ndarray, radians: float) -> np.ndarray:
    cos_v = np.cos(radians)
    sin_v = np.sin(radians)
    return np.array([vec[0] * cos_v - vec[1] * sin_v, vec[0] * sin_v + vec[1] * cos_v])


def _alpha(prob: float) -> float:
    return 0.5 + 0.5 * prob


def _draw_probability_label(
    ax,
    xy: np.ndarray,
    prob: float,
    fontsize: int,
    weight: str = "normal",
) -> None:
    ax.text(
        xy[0],
        xy[1],
        f"{prob:.0%}",
        fontsize=fontsize,
        ha="center",
        va="center",
        color="black",
        fontweight=weight,
        zorder=5,
        bbox={
            "boxstyle": "round,pad=0.06",
            "fc": "white",
            "ec": "none",
            "alpha": 0.9,
        },
    )


def _draw_cross_edge(
    ax,
    src_pos: np.ndarray,
    dst_pos: np.ndarray,
    prob: float,
    label_fontsize: int,
    curvature: float,
) -> None:
    direction = _unit(dst_pos - src_pos)
    start = src_pos + direction * NODE_RADIUS
    end = dst_pos - direction * NODE_RADIUS

    start, end = end, start

    patch = mpatches.FancyArrowPatch(
        start,
        end,
        arrowstyle="<|-",
        connectionstyle=f"arc3,rad={curvature:.2f}",
        mutation_scale=6,
        linewidth=1.0,
        edgecolor="black",
        facecolor="none",
        alpha=_alpha(prob),
        zorder=2,
    )
    ax.add_patch(patch)

    normal = np.array([-direction[1], direction[0]])
    label_pos = (start + end) / 2 + normal * curvature * np.linalg.norm(end - start)
    _draw_probability_label(ax, label_pos, prob, label_fontsize)


def _draw_self_loop(
    ax,
    node_pos: np.ndarray,
    graph_center: np.ndarray,
    prob: float,
    label_fontsize: int,
) -> None:
    outward = _unit(node_pos - graph_center)
    tangent = np.array([-outward[1], outward[0]])
    start = node_pos + _rotate(outward, -0.95) * NODE_RADIUS
    end = node_pos + _rotate(outward, 0.95) * NODE_RADIUS
    ctrl1 = node_pos + outward * 0.34 - tangent * 0.16
    ctrl2 = node_pos + outward * 0.34 + tangent * 0.16
    path = PlotPath(
        [start, ctrl1, ctrl2, end],
        [PlotPath.MOVETO, PlotPath.CURVE4, PlotPath.CURVE4, PlotPath.CURVE4],
    )
    patch = mpatches.FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=6,
        linewidth=1.0,
        edgecolor="black",
        facecolor="none",
        alpha=_alpha(prob),
        zorder=2,
    )
    ax.add_patch(patch)
    _draw_probability_label(
        ax,
        node_pos + outward * 0.33,
        prob,
        label_fontsize,
        weight="bold",
    )


def _draw_transition_edges(
    ax,
    nodes: tuple[str, ...],
    pos: dict[str, np.ndarray],
    transition_probs: dict[str, dict[str, float]],
    sources: tuple[str, ...] | None = None,
    label_fontsize: int = 6,
) -> None:
    graph_center = np.mean([pos[node] for node in nodes], axis=0)
    edge_sources = sources if sources is not None else nodes
    curvature = 0.26 if len(nodes) == 2 else 0.18

    for src in edge_sources:
        for dst in nodes:
            prob = transition_probs.get(src, {}).get(dst, 0.0)
            if src == dst:
                _draw_self_loop(ax, pos[src], graph_center, prob, label_fontsize)
            else:
                _draw_cross_edge(
                    ax,
                    pos[src],
                    pos[dst],
                    prob,
                    label_fontsize,
                    curvature,
                )


def _draw_nodes(
    ax,
    nodes: tuple[str, ...],
    pos: dict[str, np.ndarray],
    abbr_map: dict[str, str] | None = None,
) -> None:
    for node in nodes:
        x, y = pos[node]
        circle = plt.Circle(
            (x, y), NODE_RADIUS, fc="white", ec="black", lw=1.0, zorder=3
        )
        ax.add_patch(circle)
        label = abbr_map.get(node, node) if abbr_map else node
        ax.text(
            x,
            y,
            label,
            fontsize=7,
            ha="center",
            va="center",
            fontweight="bold",
            color="black",
            zorder=4,
        )


def _style_panel(ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_color("#aaaaaa")
        spine.set_linewidth(0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")


def _plot_within_day_chart(probs: dict, action: str, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(19, 7.5))
    fig.suptitle(
        (
            f"Within-day step-bin transitions after action: {action} "
            f"(averaged across {TIMESTEPS} timesteps)"
        ),
        fontsize=13,
        fontweight="bold",
    )

    for row_idx, day in enumerate(DAY_TYPES):
        axes[row_idx, 0].set_ylabel(
            f"Day type: {day}",
            fontsize=10,
            fontweight="bold",
            rotation=90,
            labelpad=12,
        )

    for col_idx, burden in enumerate(BURDENS):
        for sleep_idx, sleep in enumerate(SLEEP_TYPES):
            axes[0, col_idx * 2 + sleep_idx].set_title(
                f"Burden: {burden}\nSleep: {sleep}", fontsize=8
            )

    for row_idx, day in enumerate(DAY_TYPES):
        for col_idx, burden in enumerate(BURDENS):
            for sleep_idx, sleep in enumerate(SLEEP_TYPES):
                ax = axes[row_idx, col_idx * 2 + sleep_idx]
                _style_panel(ax, (-0.9, 0.9), (-0.8, 0.95))
                transition_probs = {
                    step_bin: probs.get((step_bin, burden, action, day, sleep), {})
                    for step_bin in STEP_BINS
                }
                _draw_transition_edges(
                    ax,
                    tuple(STEP_BINS),
                    _POS,
                    transition_probs,
                    label_fontsize=6,
                )
                _draw_nodes(ax, tuple(STEP_BINS), _POS, STEP_NODE_ABBR)

    fig.tight_layout()
    fig.savefig(fig_dir / f"{action}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_day_boundary_chart(db_probs: dict, fig_dir: Path) -> None:
    combos = list(itertools.product(BURDENS, DAY_TYPES, SLEEP_TYPES))
    nrows = 6
    ncols = 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 18))
    fig.suptitle(
        (
            "Day-boundary sleep transitions. Each panel fixes daily step bin, "
            "burden, day type, and current sleep; arrows show P(next sleep)."
        ),
        fontsize=13,
        fontweight="bold",
    )

    for row_idx, step_bin in enumerate(STEP_BINS):
        for col_idx, (burden, day, sleep) in enumerate(combos):
            flat_idx = row_idx * len(combos) + col_idx
            ax = axes[flat_idx // ncols, flat_idx % ncols]
            if col_idx == 0:
                ax.set_ylabel(
                    f"Daily steps: {step_bin}",
                    fontsize=9,
                    fontweight="bold",
                    rotation=90,
                    labelpad=8,
                )
            ax.set_title(
                (
                    f"Steps: {step_bin}\n"
                    f"Burden: {burden} | Day: {day}\n"
                    f"Current sleep: {sleep}"
                ),
                fontsize=7,
            )

            _style_panel(ax, (-0.9, 0.9), (-0.65, 0.55))
            transition_probs = {sleep: db_probs.get((step_bin, burden, day, sleep), {})}
            _draw_transition_edges(
                ax,
                tuple(SLEEP_TYPES),
                _SLEEP_POS,
                transition_probs,
                sources=(sleep,),
                label_fontsize=5,
            )
            _draw_nodes(
                ax,
                tuple(SLEEP_TYPES),
                _SLEEP_POS,
                {"good": "g", "poor": "p"},
            )

    total_cells = len(STEP_BINS) * len(combos)
    for idx in range(total_cells, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.4, w_pad=0.8)
    fig.savefig(fig_dir / "day_boundary.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Results JSONL files to process",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Base directory for figure output",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("tables"),
        help="Directory for aggregated JSON tables",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for fpath in args.files:
        path = Path(fpath)
        label = model_label(path)
        logger.info("")
        logger.info("Processing %s...", label)

        records = load(path)
        logger.info("  Loaded %s records", len(records))

        logger.info("  Aggregating day-boundary transitions...")
        db_probs = aggregate_day_boundary(records)
        logger.info("  %s unique day-boundary cells", len(db_probs))

        logger.info("  Aggregating within-day transitions (per-timestep)...")
        wd_probs = aggregate_within_day(records)
        total_wd = sum(len(value) for value in wd_probs.values())
        logger.info(
            "  %s unique within-day cells across %s timesteps",
            total_wd,
            TIMESTEPS,
        )

        logger.info("  Averaging within-day across timesteps...")
        wd_avg = aggregate_within_day_averaged(records)
        logger.info("  %s unique averaged within-day cells", len(wd_avg))

        tables_dir = args.tables_dir / label
        _save_tables(db_probs, wd_probs, tables_dir)

        fig_dir = args.fig_dir / label / "matrices"
        fig_dir.mkdir(parents=True, exist_ok=True)
        logger.info("  Generating figures in %s/", fig_dir)

        for action in ACTIONS:
            _plot_within_day_chart(wd_avg, action, fig_dir)
        _plot_day_boundary_chart(db_probs, fig_dir)

        logger.info("  Done: 5 charts written (4 actions + 1 day_boundary)")

    logger.info("")
    logger.info("%s", "=" * 60)
    logger.info("  All done.")
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
