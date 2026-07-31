"""Validate a PEARL transition table by running it through the simulator.

Loads the table, checks action coverage and intervention direction, and
runs one 60-day episode per arm through the simulator to confirm the
pilot table is consumable end-to-end.
"""
# ruff: noqa: E402, PLC0415

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl_health_interventions.llm_bootstrapping._shared import (
    setup_logging,
)

logger = logging.getLogger(__name__)

_DEFAULT_TABLE = _REPO_ROOT / "tables" / "pearl_12action_pilot" / "pearl_pilot.json"
_DEFAULT_CONFIG = (
    _REPO_ROOT
    / "docs"
    / "experimental_phases"
    / "pearl_random"
    / "configs"
    / "pearl_bootstrap.yaml"
)


def check_intervention_direction(table: dict) -> None:
    """Log ΔP(high)/ΔP(low) for ability_morning vs idle per full state."""
    groups: dict[str, dict[str, dict]] = {}
    for t in table["transitions"]:
        state_key = json.dumps(t["state"], sort_keys=True)
        groups.setdefault(state_key, {})[t["action"]] = t["next_state_probs"]

    logger.info("\n=== Intervention Effect Analysis ===")
    for state_key, action_probs in sorted(groups.items()):
        state = json.loads(state_key)
        idle = action_probs.get("idle")
        ability = action_probs.get("ability_morning")
        if idle is None or ability is None:
            continue
        idle_probs = idle.get("recent_steps_mean", {})
        ability_probs = ability.get("recent_steps_mean", {})
        diff_high = ability_probs.get("high", 0.0) - idle_probs.get("high", 0.0)
        diff_low = ability_probs.get("low", 0.0) - idle_probs.get("low", 0.0)
        logger.info(
            "  %s / burden=%s: ΔP(high)=%+.3f ΔP(low)=%+.3f %s",
            state["recent_steps_mean"],
            state["burden"],
            diff_high,
            diff_low,
            "✓" if diff_high > 0 else "✗ WRONG DIRECTION",
        )


def run_simulator_smoke_test(table_path: Path, config_path: Path) -> None:
    """Run one 60-day episode per arm against the pilot table.

    The transition loader reads every ``*.json`` in the table directory, so
    the target table is copied to a temp directory to keep the test isolated.
    """
    from rl_health_interventions.config.loader import load_config
    from scripts.pearl_constitution.utils import (
        compute_arm_daily_steps,
        run_all_arms,
    )

    config = load_config(str(config_path))
    with tempfile.TemporaryDirectory() as tmp_dir:
        table_dir = Path(tmp_dir)
        shutil.copy2(table_path, table_dir / table_path.name)
        config.transition_model.table_dir = str(table_dir)

        logger.info("\n=== Simulator Test ===")
        trajectories = run_all_arms(config, n_seeds=1)

    daily_steps = compute_arm_daily_steps(trajectories)
    for arm, steps in daily_steps.items():
        logger.info("  %s: mean daily steps = %.1f", arm, float(steps.mean()))
    logger.info("Simulation completed successfully")


def main() -> None:
    """Load a pilot table, check action coverage and intervention direction."""
    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", nargs="?", type=Path, default=_DEFAULT_TABLE)
    parser.add_argument("config", nargs="?", type=Path, default=_DEFAULT_CONFIG)
    args = parser.parse_args()

    logger.info("Loading table from %s", args.table)
    logger.info("Using config from %s", args.config)

    with args.table.open() as f:
        table = json.load(f)

    n_transitions = len(table["transitions"])
    logger.info("Table has %d transitions", n_transitions)

    action_counts = {}
    for t in table["transitions"]:
        action = t["action"]
        action_counts[action] = action_counts.get(action, 0) + 1
    logger.info("Action coverage: %d/%d actions", len(action_counts), 13)

    check_intervention_direction(table)
    run_simulator_smoke_test(args.table, args.config)


if __name__ == "__main__":
    main()
