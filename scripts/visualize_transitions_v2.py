"""Black-and-white NetworkX transition matrix charts.

B/W styling with visible self-loops, probability labels on all edges,
and subplot borders.  Produces 5 charts per model (4 actions + 1 day_boundary),
aggregated across the 5 within-day timesteps.

Usage:
    uv run python scripts/visualize_transitions_v2.py \\
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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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

STEP_NODE_ABBR = {"inactive": "i", "moderate": "m", "active": "a"}


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


def aggregate_within_day_averaged(records: list[dict]) -> dict:
    """Average transition probabilities across all 5 within-day timesteps."""
    per_t = aggregate_within_day(records)
    if not per_t:
        return {}

    all_keys = set()
    for t_probs in per_t.values():
        all_keys.update(t_probs.keys())

    averaged: dict = {}
    for key in all_keys:
        all_outcomes: dict[str, list[float]] = defaultdict(list)
        count = 0
        for t in range(TIMESTEPS):
            if key in per_t.get(t, {}):
                for outcome, prob in per_t[t][key].items():
                    all_outcomes[outcome].append(prob)
                count += 1
        if count > 0:
            averaged[key] = {
                outcome: sum(probs) / count for outcome, probs in all_outcomes.items()
            }
    return averaged


def _save_tables(db_probs: dict, wd_probs: dict[int, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {}
    for key, outcomes in db_probs.items():
        state_key = "|".join(str(v) for v in key)
        table[state_key] = {"_": outcomes}
    with open(out_dir / "day_boundary.json", "w") as f:
        json.dump(table, f, indent=2)
    for t in range(TIMESTEPS):
        table = {}
        for key, outcomes in wd_probs.get(t, {}).items():
            state_key = "|".join(str(v) for v in key)
            table[state_key] = {"_": outcomes}
        with open(out_dir / f"within_day_{t}.json", "w") as f:
            json.dump(table, f, indent=2)
    print(f"  Wrote 6 tables to {out_dir}/")


# Fixed node positions for the 3-node step_bin triangle
_POS = {
    "inactive": np.array([-0.45, -0.26]),
    "moderate": np.array([0.0, 0.52]),
    "active": np.array([0.45, -0.26]),
}

# 2-node positions for day-boundary sleep graph
_SLEEP_POS = {"good": np.array([-0.4, 0.0]), "poor": np.array([0.4, 0.0])}


def _draw_edges(ax, nodes, pos, probs, label_fontsize=6):
    """Draw directed edges with probability labels on a B/W graph.

    - Fixed line width (lw=1.0) for all edges.
    - Alpha scales with probability: 0.5 at 0%, 1.0 at 100%.
    - Self-loops use FancyArrowPatch so they originate and terminate on the
      same node.
    """
    for src in nodes:
        for dst in nodes:
            p = probs.get(dst, 0.0)
            if p < 0.01:
                continue
            alpha = 0.5 + 0.5 * p
            src_pos = pos[src]
            dst_pos = pos[dst]

            if src == dst:
                # Self-loop: FancyArrowPatch from node back to itself
                loop = mpatches.FancyArrowPatch(
                    src_pos,
                    src_pos,
                    connectionstyle="arc3,rad=0.5",
                    arrowstyle="-|>",
                    mutation_scale=6,
                    linewidth=1.0,
                    edgecolor="black",
                    facecolor="none",
                    alpha=alpha,
                    zorder=2,
                )
                ax.add_patch(loop)
                # Label: offset to the right of the loop (rad>0 arcs right)
                label_pos = src_pos + np.array([0.32, 0.0])
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    f"{p:.0%}",
                    fontsize=label_fontsize,
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.06",
                        "fc": "white",
                        "ec": "none",
                        "alpha": 0.9,
                    },
                )
            else:
                # Cross-edge: curved arrow with label
                has_reverse = probs.get(src, 0.0) > 0.01
                rad = -0.25 if has_reverse else 0.0
                style = f"arc3,rad={rad:.2f}"
                mid = (src_pos + dst_pos) / 2
                normal = dst_pos - src_pos
                normal = np.array([-normal[1], normal[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-9)
                label_offset = normal * 0.08 * (1 if rad > 0 else -1)
                ax.annotate(
                    "",
                    xy=dst_pos,
                    xytext=src_pos,
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "black",
                        "lw": 1.0,
                        "alpha": alpha,
                        "connectionstyle": style,
                        "mutation_scale": 6,
                    },
                )
                ax.text(
                    mid[0] + label_offset[0],
                    mid[1] + label_offset[1],
                    f"{p:.0%}",
                    fontsize=label_fontsize,
                    ha="center",
                    va="center",
                    color="black",
                    bbox={
                        "boxstyle": "round,pad=0.06",
                        "fc": "white",
                        "ec": "none",
                        "alpha": 0.9,
                    },
                )


def _draw_nodes(ax, nodes, pos, abbr_map=None):
    """Draw B/W nodes with black outline and white fill."""
    for node in nodes:
        x, y = pos[node]
        circle = plt.Circle((x, y), 0.1, fc="white", ec="black", lw=1.0, zorder=3)
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


def _plot_within_day_chart(
    probs: dict,
    action: str,
    fig_dir: Path,
) -> None:
    """Plot within-day transitions for one action, averaged across timesteps."""
    fig, axes = plt.subplots(2, 6, figsize=(18, 7))
    fig.suptitle(
        f"action={action}  (avg over {TIMESTEPS} timesteps)",
        fontsize=14,
        fontweight="bold",
    )

    # Row headers (day_of_week)
    for row_idx, day in enumerate(DAY_TYPES):
        axes[row_idx, 0].set_ylabel(
            day, fontsize=10, fontweight="bold", rotation=90, labelpad=12
        )

    # Column headers (burden / sleep)
    for col_idx, burden in enumerate(BURDENS):
        for sleep_idx, sleep in enumerate(SLEEP_TYPES):
            c = col_idx * 2 + sleep_idx
            axes[0, c].set_title(f"{burden}/{sleep[:3]}", fontsize=8)

    for row_idx, day in enumerate(DAY_TYPES):
        for col_idx, burden in enumerate(BURDENS):
            for sleep_idx, sleep in enumerate(SLEEP_TYPES):
                c = col_idx * 2 + sleep_idx
                ax = axes[row_idx, c]
                ax.set_frame_on(True)
                for spine in ax.spines.values():
                    spine.set_color("#aaaaaa")
                    spine.set_linewidth(0.5)
                ax.tick_params(
                    left=False, bottom=False, labelleft=False, labelbottom=False
                )
                ax.set_xlim(-0.85, 0.85)
                ax.set_ylim(-0.65, 0.85)
                ax.set_aspect("equal")

                # Build transition probs for each current step_bin
                all_probs: dict[str, dict[str, float]] = {}
                for sb in STEP_BINS:
                    key = (sb, burden, action, day, sleep)
                    all_probs[sb] = probs.get(key, {})

                _draw_nodes(ax, STEP_BINS, _POS, STEP_NODE_ABBR)
                for src in STEP_BINS:
                    _draw_edges(ax, STEP_BINS, _POS, all_probs[src])

    fig.tight_layout()
    fig.savefig(
        fig_dir / f"{action}_v2.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_day_boundary_chart(db_probs: dict, fig_dir: Path) -> None:
    combos = list(itertools.product(BURDENS, DAY_TYPES, SLEEP_TYPES))
    nrows = 4
    ncols = 9
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 10))
    fig.suptitle(
        "day_boundary  P(sleep' | step_bin_daily, burden, day, sleep)",
        fontsize=14,
        fontweight="bold",
    )

    for row_idx, sb in enumerate(STEP_BINS):
        for col_idx, (burden, day, sleep) in enumerate(combos):
            flat_idx = row_idx * 12 + col_idx
            r = flat_idx // ncols
            c = flat_idx % ncols
            ax = axes[r, c]
            key = (sb, burden, day, sleep)
            outcomes = db_probs.get(key, {})

            if col_idx == 0:
                ax.set_ylabel(
                    sb, fontsize=8, fontweight="bold", rotation=90, labelpad=8
                )
            if row_idx == 0:
                ax.set_title(f"{burden[:2]}/{day[:1]}/{sleep[:1]}", fontsize=6)

            ax.set_frame_on(True)
            for spine in ax.spines.values():
                spine.set_color("#aaaaaa")
                spine.set_linewidth(0.5)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_xlim(-0.75, 0.75)
            ax.set_ylim(-0.5, 0.5)
            ax.set_aspect("equal")

            _draw_nodes(
                ax,
                ("good", "poor"),
                _SLEEP_POS,
                {"good": "g", "poor": "p"},
            )
            _draw_edges(
                ax,
                ("good", "poor"),
                _SLEEP_POS,
                outcomes,
                label_fontsize=5,
            )

    total_cells = len(STEP_BINS) * len(combos)
    for idx in range(total_cells, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(fig_dir / "day_boundary_v2.png", dpi=150, bbox_inches="tight")
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

        print("  Aggregating day-boundary transitions...")
        db_probs = aggregate_day_boundary(records)
        print(f"  {len(db_probs)} unique day-boundary cells")

        print("  Aggregating within-day transitions (per-timestep)...")
        wd_probs = aggregate_within_day(records)
        total_wd = sum(len(v) for v in wd_probs.values())
        print(f"  {total_wd} unique within-day cells across {TIMESTEPS} timesteps")

        print("  Averaging within-day across timesteps...")
        wd_avg = aggregate_within_day_averaged(records)
        print(f"  {len(wd_avg)} unique averaged within-day cells")

        tables_dir = args.tables_dir / label
        _save_tables(db_probs, wd_probs, tables_dir)

        fig_dir = args.fig_dir / label / "matricies"
        fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Generating figures in {fig_dir}/")

        for action in ACTIONS:
            _plot_within_day_chart(wd_avg, action, fig_dir)

        _plot_day_boundary_chart(db_probs, fig_dir)

        print("  Done: 5 charts written (4 actions + 1 day_boundary)")

    print(f"\n{'=' * 60}")
    print("  All done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
