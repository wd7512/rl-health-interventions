"""Generate PEARL 12-action transition tables.

Produces per-factor JSON tables at tables/pearl_12action/ for use with the
table_transition transition model.  Run once, commit output:

    uv run python docs/experimental_phases/pearl_random/generate_tables.py
"""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIG_PATH = (
    _REPO_ROOT
    / "docs"
    / "experimental_phases"
    / "pearl_random"
    / "configs"
    / "pearl_random.yaml"
)
_OUTPUT_DIR = _REPO_ROOT / "tables" / "pearl_12action"


def _generate_tables(config):  # noqa: ANN001
    """Generate per-(factor_value, action) Dirichlet tables."""
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(config.seed)

    stochastic = [
        n for n, c in config.state.variables.items() if c.advanced is None
    ]
    actions = list(config.actions.keys())

    # day_boundary: one Dirichlet draw per (factor, factor_value)
    day_boundary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for name in stochastic:
        var_cfg = config.state.variables[name]
        day_boundary[name] = {}
        for fv in var_cfg.names:
            probs = rng.dirichlet([1.0] * len(var_cfg.names))
            day_boundary[name][fv] = {
                "_": {t: float(p) for t, p in zip(var_cfg.names, probs, strict=True)}
            }

    # within_day: one Dirichlet draw per (factor, factor_value, action)
    within_day: list[dict[str, dict[str, dict[str, dict[str, float]]]]] = []
    for _step_idx in range(config.steps_per_day):
        step_tables: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for name in stochastic:
            var_cfg = config.state.variables[name]
            step_tables[name] = {}
            for fv in var_cfg.names:
                for action in actions:
                    probs = rng.dirichlet([1.0] * len(var_cfg.names))
                    step_tables[name][f"{fv}|{action}"] = {
                        "_": {
                            t: float(p)
                            for t, p in zip(var_cfg.names, probs, strict=True)
                        }
                    }
        within_day.append(step_tables)

    return day_boundary, within_day


def _save_per_factor_tables(
    day_boundary, within_day, output_dir, config  # noqa: ANN001
) -> None:
    """Save tables in per-factor format with _format marker."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # day_boundary.json
    db_path = output_dir / "day_boundary.json"
    db_data = {"_format": "per_factor"}
    db_data.update(day_boundary)
    with db_path.open("w", encoding="utf-8") as f:
        json.dump(db_data, f, indent=2)
    logger.info("Saved day_boundary.json (per_factor format)")

    # within_day_N.json
    for step_idx, step_tables in enumerate(within_day):
        wd_path = output_dir / f"within_day_{step_idx}.json"
        wd_data = {"_format": "per_factor"}
        wd_data.update(step_tables)
        with wd_path.open("w", encoding="utf-8") as f:
            json.dump(wd_data, f, indent=2)
        logger.info("Saved within_day_%d.json (per_factor format)", step_idx)


def main() -> None:
    from rl_health_interventions.config.loader import load_config  # noqa: PLC0415

    logger.info("Loading config from %s", _CONFIG_PATH)
    config = load_config(_CONFIG_PATH)

    stochastic = [
        n for n, c in config.state.variables.items() if c.advanced is None
    ]
    actions = list(config.actions.keys())
    logger.info(
        "Config loaded: %d stochastic factors, %d actions, %d steps_per_day",
        len(stochastic),
        len(actions),
        config.steps_per_day,
    )

    logger.info("Generating tables with seed=%d", config.seed)
    day_boundary_tables, within_day_tables = _generate_tables(config)

    _save_per_factor_tables(
        day_boundary_tables, within_day_tables, _OUTPUT_DIR, config
    )
    logger.info("Done. Tables saved to %s", _OUTPUT_DIR)


if __name__ == "__main__":
    main()
