"""Aggregate bootstrap JSONL and produce NetworkX transition matrix charts.

For each model: aggregates raw LLM responses into 6 JSON transition table
files, then produces 21 NetworkX directed graph figures (5 within-day x 4
actions + 1 day-boundary).

Usage:
    uv run python scripts/visualize_transitions.py \\
        --files data/bootstrap/results_deepseek.jsonl \\
               data/bootstrap/results_glm5.2.jsonl \\
        --fig-dir docs/figures
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_health_interventions.llm_bootstrapping.parse import (
    parse_day_boundary,
    parse_within_day,
)
from rl_health_interventions.llm_bootstrapping.prompts.sprint1 import (
    ACTIONS,
    BURDENS,
    DAY_TYPES,
    SLEEP_TYPES,
    STEP_BINS,
)

logger = logging.getLogger(__name__)

SAMPLES_DAY_BOUNDARY = 20
SAMPLES_WITHIN_DAY = 30
WITHIN_DAY_START = 720
WITHIN_DAY_CELLS_PER_TIMESTEP = 144
TIMESTEPS = 5

STEP_BIN_COLORS = {"inactive": "#e74c3c", "moderate": "#f39c12", "active": "#2ecc71"}
SLEEP_COLORS = {"good": "#2ecc71", "poor": "#e74c3c"}

# Curved triangle node positions for the 3-node step_bin graph
_TRIANGLE_POS = {
    "inactive": np.array([-0.5, -0.288]),
    "moderate": np.array([0.0, 0.577]),
    "active": np.array([0.5, -0.288]),
}

# 2-node positions for day-boundary sleep graph
_SLEEP_POS = {"good": np.array([-0.5, 0.0]), "poor": np.array([0.5, 0.0])}


def _day_boundary_params(cell_idx: int) -> dict:
    combos = list(itertools.product(STEP_BINS, BURDENS, DAY_TYPES, SLEEP_TYPES))
    sb, b, d, s = combos[cell_idx]
    return {"step_bin_daily": sb, "burden": b, "day_type": d, "sleep": s}


def _within_day_params(cell_idx: int) -> dict:
    combos = list(
        itertools.product(STEP_BINS, BURDENS, ACTIONS, DAY_TYPES, SLEEP_TYPES)
    )
    sb, b, a, d, s = combos[cell_idx]
    return {"step_bin": sb, "burden": b, "action": a, "day_type": d, "sleep": s}


def _parse_record(rec: dict) -> tuple[str, object]:
    idx = rec["index"]
    content = rec.get("content", "")
    if idx < WITHIN_DAY_START:
        return "day_boundary", parse_day_boundary(content)
    return "within_day", parse_within_day(content)


def _load(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def aggregate_day_boundary(records: list[dict]) -> dict:
    """Aggregate day-boundary responses into P(sleep' | state) per cell."""
    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[tuple, int] = defaultdict(int)
    for rec in records:
        if rec["index"] >= WITHIN_DAY_START:
            continue
        cat, parsed = _parse_record(rec)
        if cat != "day_boundary" or parsed is None:
            continue
        cell_idx = rec["index"] // SAMPLES_DAY_BOUNDARY
        params = _day_boundary_params(cell_idx)
        key = (
            params["step_bin_daily"],
            params["burden"],
            params["day_type"],
            params["sleep"],
        )
        counts[key][parsed] += 1
        totals[key] += 1
    probs = {}
    for key, outcomes in counts.items():
        t = totals[key]
        probs[key] = {o: c / t for o, c in outcomes.items()}
    return probs


def aggregate_within_day(records: list[dict]) -> dict[int, dict]:
    """Aggregate within-day responses into P(step_bin'|state) per cell."""
    all_probs: dict[int, dict] = {}
    for t in range(TIMESTEPS):
        counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        totals: dict[tuple, int] = defaultdict(int)
        for rec in records:
            if rec["index"] < WITHIN_DAY_START:
                continue
            offset = rec["index"] - WITHIN_DAY_START
            timestep = offset // (WITHIN_DAY_CELLS_PER_TIMESTEP * SAMPLES_WITHIN_DAY)
            if timestep != t:
                continue
            within_off = offset % (WITHIN_DAY_CELLS_PER_TIMESTEP * SAMPLES_WITHIN_DAY)
            cell_idx = within_off // SAMPLES_WITHIN_DAY
            params = _within_day_params(cell_idx)
            key = (
                params["step_bin"],
                params["burden"],
                params["action"],
                params["day_type"],
                params["sleep"],
            )
            cat, parsed = _parse_record(rec)
            if cat != "within_day" or parsed is None:
                continue
            _, sb = parsed
            counts[key][sb] += 1
            totals[key] += 1
        probs = {}
        for key, outcomes in counts.items():
            tt = totals[key]
            probs[key] = {o: c / tt for o, c in outcomes.items()}
        all_probs[t] = probs
    return all_probs


def _save_tables(db_probs: dict, wd_probs: dict[int, dict], out_dir: Path) -> None:
    """Write 6 JSON table files matching the decisions doc format."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Day-boundary table
    table = {}
    for key, outcomes in db_probs.items():
        state_key = "|".join(str(v) for v in key)
        table[state_key] = {"_": outcomes}
    with open(out_dir / "day_boundary.json", "w") as f:
        json.dump(table, f, indent=2)

    # Within-day tables
    for t in range(TIMESTEPS):
        table = {}
        for key, outcomes in wd_probs.get(t, {}).items():
            state_key = "|".join(str(v) for v in key)
            table[state_key] = {"_": outcomes}
        with open(out_dir / f"within_day_{t}.json", "w") as f:
            json.dump(table, f, indent=2)

    print(f"  Wrote 6 tables to {out_dir}/")


def _draw_step_bin_graph(ax: nx.DiGraph, probs: dict[str, float]) -> None:
    """Draw a 3-node directed graph on the given axes."""
    ax.set_aspect("equal")
    ax.axis("off")
    g = nx.DiGraph()
    for node in STEP_BINS:
        g.add_node(node)
    for src in STEP_BINS:
        for dst in STEP_BINS:
            p = probs.get(dst, 0.0)
            if p > 0.01:
                g.add_edge(src, dst, weight=p)

    node_colors = [STEP_BIN_COLORS[n] for n in g.nodes]
    node_sizes = [800 for _ in g.nodes]
    nx.draw_networkx_nodes(
        g,
        _TRIANGLE_POS,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=0.8,
    )
    nx.draw_networkx_labels(
        g,
        _TRIANGLE_POS,
        ax=ax,
        font_size=5,
        font_color="white",
        font_weight="bold",
    )

    edge_data = g.edges(data=True)
    if edge_data:
        weights = [d["weight"] for _, _, d in edge_data]
        max_w = max(weights) if weights else 1.0
        for u, v, d in edge_data:
            w = d["weight"]
            rad = 0.2 if u != v else 0.4
            style = f"arc3,rad={rad:.2f}"
            color = STEP_BIN_COLORS.get(u, "#666666")
            width = 0.5 + 3.5 * (w / max_w)
            ax.annotate(
                "",
                xy=_TRIANGLE_POS[v],
                xytext=_TRIANGLE_POS[u],
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": width,
                    "connectionstyle": style,
                    "mutation_scale": 8,
                    "alpha": 0.7,
                },
            )


def _draw_sleep_graph(ax: nx.DiGraph, probs: dict[str, float]) -> None:
    """Draw a 2-node directed graph for sleep transitions."""
    ax.set_aspect("equal")
    ax.axis("off")
    g = nx.DiGraph()
    g.add_node("good")
    g.add_node("poor")
    for src in ("good", "poor"):
        for dst in ("good", "poor"):
            p = probs.get(dst, 0.0)
            if p > 0.01:
                g.add_edge(src, dst, weight=p)

    node_colors = [SLEEP_COLORS[n] for n in g.nodes]
    nx.draw_networkx_nodes(
        g,
        _SLEEP_POS,
        ax=ax,
        node_color=node_colors,
        node_size=700,
        edgecolors="black",
        linewidths=0.8,
    )
    nx.draw_networkx_labels(
        g,
        _SLEEP_POS,
        ax=ax,
        font_size=6,
        font_color="white",
        font_weight="bold",
    )

    edge_data = g.edges(data=True)
    if edge_data:
        weights = [d["weight"] for _, _, d in edge_data]
        max_w = max(weights) if weights else 1.0
        for u, v, d in edge_data:
            w = d["weight"]
            rad = 0.3 if u != v else 0.0
            style = f"arc3,rad={rad:.2f}"
            color = SLEEP_COLORS.get(u, "#666666")
            width = 0.5 + 3.5 * (w / max_w)
            ax.annotate(
                "",
                xy=_SLEEP_POS[v],
                xytext=_SLEEP_POS[u],
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": width,
                    "connectionstyle": style,
                    "mutation_scale": 8,
                    "alpha": 0.7,
                },
            )


def _plot_within_day_chart(
    probs: dict[int, dict],
    action: str,
    timestep: int,
    fig_dir: Path,
) -> None:
    """One chart: 2 rows (day_of_week) x 6 cols (burden x sleep)."""
    fig, axes = plt.subplots(2, 6, figsize=(15, 6))
    fig.suptitle(
        f"t={timestep}  action={action}",
        fontsize=14,
        fontweight="bold",
    )

    # Row headers
    for row_idx, day in enumerate(DAY_TYPES):
        axes[row_idx, 0].set_ylabel(day, fontsize=10, fontweight="bold")

    # Column headers
    for col_idx, burden in enumerate(BURDENS):
        for sleep_idx, sleep in enumerate(SLEEP_TYPES):
            c = col_idx * 2 + sleep_idx
            axes[0, c].set_title(
                f"{burden}/{sleep[:3]}",
                fontsize=8,
            )

    timestep_probs = probs.get(timestep, {})
    for row_idx, day in enumerate(DAY_TYPES):
        for col_idx, burden in enumerate(BURDENS):
            for sleep_idx, sleep in enumerate(SLEEP_TYPES):
                c = col_idx * 2 + sleep_idx
                ax = axes[row_idx, c]
                # Build probs dict for current step_bin
                edge_probs: dict[str, dict[str, float]] = {}
                for sb in STEP_BINS:
                    full_key = (sb, burden, action, day, sleep)
                    outcomes = timestep_probs.get(full_key, {})
                    edge_probs[sb] = outcomes
                # For the graph, we need P(next | current) for each current.
                # The graph shows all 3 nodes with edges from each current
                # to next. We need to create a merged view.
                # Use the first non-empty current step_bin to drive the graph,
                # or average over all currents.
                # Actually: show transitions FROM each node. The graph needs
                # edges from each src to each dst weighted by P(dst | src).
                g = nx.DiGraph()
                for src in STEP_BINS:
                    full_key = (src, burden, action, day, sleep)
                    outcomes = timestep_probs.get(full_key, {})
                    for dst in STEP_BINS:
                        p = outcomes.get(dst, 0.0)
                        if p > 0.01:
                            g.add_edge(src, dst, weight=p)
                node_colors = [STEP_BIN_COLORS[n] for n in g.nodes]
                nx.draw_networkx_nodes(
                    g,
                    _TRIANGLE_POS,
                    ax=ax,
                    node_color=node_colors,
                    node_size=350,
                    edgecolors="black",
                    linewidths=0.5,
                )
                nx.draw_networkx_labels(
                    g,
                    _TRIANGLE_POS,
                    ax=ax,
                    font_size=4,
                    font_color="white",
                    font_weight="bold",
                )
                edge_data = g.edges(data=True)
                if edge_data:
                    weights = [d["weight"] for _, _, d in edge_data]
                    max_w = max(weights) if weights else 1.0
                    for u, v, d in edge_data:
                        w = d["weight"]
                        rad = 0.2 if u != v else 0.4
                        style = f"arc3,rad={rad:.2f}"
                        color = STEP_BIN_COLORS.get(u, "#666666")
                        width = 0.3 + 2.0 * (w / max_w)
                        ax.annotate(
                            "",
                            xy=_TRIANGLE_POS[v],
                            xytext=_TRIANGLE_POS[u],
                            arrowprops={
                                "arrowstyle": "-|>",
                                "color": color,
                                "lw": width,
                                "connectionstyle": style,
                                "mutation_scale": 6,
                                "alpha": 0.7,
                            },
                        )
                ax.set_aspect("equal")
                ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        fig_dir / f"{action}_t{timestep}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_day_boundary_chart(db_probs: dict, fig_dir: Path) -> None:
    """One chart: 3 rows (step_bin_daily) x 12 cols (burden x day x sleep),
    tiled as 4x9 = 36 subplots."""
    combos = list(itertools.product(BURDENS, DAY_TYPES, SLEEP_TYPES))
    nrows = 4
    ncols = 9
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 9))
    fig.suptitle(
        "day_boundary  P(sleep' | step_bin_daily, burden, day, sleep)",
        fontsize=14,
        fontweight="bold",
    )

    # Flatten the 3 x 12 grid into 36 subplot positions
    for row_idx, sb in enumerate(STEP_BINS):
        for col_idx, (burden, day, sleep) in enumerate(combos):
            flat_idx = row_idx * 12 + col_idx
            r = flat_idx // ncols
            c = flat_idx % ncols
            ax = axes[r, c]
            key = (sb, burden, day, sleep)
            outcomes = db_probs.get(key, {})

            # Row label on first column of each step_bin row
            if col_idx == 0:
                ax.set_ylabel(sb, fontsize=8, fontweight="bold")

            # Column label on top row
            if row_idx == 0:
                ax.set_title(f"{burden[:2]}/{day[:1]}/{sleep[:1]}", fontsize=6)

            g = nx.DiGraph()
            g.add_node("good")
            g.add_node("poor")
            for src in ("good", "poor"):
                for dst in ("good", "poor"):
                    p = outcomes.get(dst, 0.0)
                    if p > 0.01:
                        g.add_edge(src, dst, weight=p)

            node_colors = [SLEEP_COLORS[n] for n in g.nodes]
            nx.draw_networkx_nodes(
                g,
                _SLEEP_POS,
                ax=ax,
                node_color=node_colors,
                node_size=300,
                edgecolors="black",
                linewidths=0.5,
            )
            nx.draw_networkx_labels(
                g,
                _SLEEP_POS,
                ax=ax,
                font_size=4,
                font_color="white",
                font_weight="bold",
            )

            edge_data = g.edges(data=True)
            if edge_data:
                weights = [d["weight"] for _, _, d in edge_data]
                max_w = max(weights) if weights else 1.0
                for u, v, d in edge_data:
                    w = d["weight"]
                    rad = 0.3 if u != v else 0.0
                    style = f"arc3,rad={rad:.2f}"
                    color = SLEEP_COLORS.get(u, "#666666")
                    width = 0.3 + 2.0 * (w / max_w)
                    ax.annotate(
                        "",
                        xy=_SLEEP_POS[v],
                        xytext=_SLEEP_POS[u],
                        arrowprops={
                            "arrowstyle": "-|>",
                            "color": color,
                            "lw": width,
                            "connectionstyle": style,
                            "mutation_scale": 6,
                            "alpha": 0.7,
                        },
                    )
            ax.set_aspect("equal")
            ax.axis("off")

    # Hide unused subplot slots
    total_cells = len(STEP_BINS) * len(combos)
    for idx in range(total_cells, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(fig_dir / "day_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _model_label(path: Path) -> str:
    name = path.stem
    if name.startswith("results_"):
        return name[len("results_") :]
    return name


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

    logging.basicConfig(level=logging.WARNING)

    for fpath in args.files:
        path = Path(fpath)
        label = _model_label(path)
        print(f"\nProcessing {label}...")

        records = _load(path)
        print(f"  Loaded {len(records)} records")

        # Aggregate
        print("  Aggregating day-boundary transitions...")
        db_probs = aggregate_day_boundary(records)
        print(f"  {len(db_probs)} unique day-boundary cells")

        print("  Aggregating within-day transitions...")
        wd_probs = aggregate_within_day(records)
        total_wd = sum(len(v) for v in wd_probs.values())
        print(f"  {total_wd} unique within-day cells across {TIMESTEPS} timesteps")

        # Save tables
        tables_dir = args.tables_dir / label
        _save_tables(db_probs, wd_probs, tables_dir)

        # Generate figures
        fig_dir = args.fig_dir / label / "matrices"
        fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Generating figures in {fig_dir}/")

        # Within-day charts (5 timesteps x 4 actions = 20)
        for t in range(TIMESTEPS):
            for action in ACTIONS:
                _plot_within_day_chart(wd_probs, action, t, fig_dir)

        # Day-boundary chart (1)
        _plot_day_boundary_chart(db_probs, fig_dir)

        print("  Done: 21 charts written")

    print(f"\n{'=' * 60}")
    print("  All done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
