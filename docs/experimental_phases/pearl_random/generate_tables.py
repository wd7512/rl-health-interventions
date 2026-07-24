"""Generate PEARL 12-action transition tables.

Produces JSON tables at tables/pearl_12action/ for use with the
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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


def _flatten_to_bootstrap_format(
    tables,  # noqa: ANN001
    stochastic,  # noqa: ANN001
    actions,  # noqa: ANN001
    config,  # noqa: ANN001
):
    """Convert per-factor tables to flat bootstrap-compatible JSON format."""
    factor_value_lists = []
    for name in stochastic:
        var_cfg = config.state.variables.get(name)
        if var_cfg is None:
            continue
        factor_value_lists.append(var_cfg.names)

    # day_boundary: key = "{fv1}|{fv2}|..."
    flat_db: dict[str, dict] = {}
    for combo in itertools.product(*factor_value_lists):
        key = "|".join(combo)
        first_name = stochastic[0]
        first_val = combo[0]
        if first_name in tables and first_val in tables[first_name]:
            flat_db[key] = tables[first_name][first_val]
    return flat_db


def _flatten_within_day(
    within_day_tables,  # noqa: ANN001
    stochastic,  # noqa: ANN001
    actions,  # noqa: ANN001
    config,  # noqa: ANN001
):
    """Convert per-factor within_day tables to flat bootstrap-compatible format."""
    factor_value_lists = []
    for name in stochastic:
        var_cfg = config.state.variables.get(name)
        if var_cfg is None:
            continue
        factor_value_lists.append(var_cfg.names)

    result = []
    for step_tables in within_day_tables:
        flat: dict[str, dict] = {}
        for combo in itertools.product(*factor_value_lists):
            for action in actions:
                full_key = "|".join(combo) + "|" + action
                for name, fv in zip(stochastic, combo, strict=True):
                    sa_key = f"{fv}|{action}"
                    if name in step_tables and sa_key in step_tables[name]:
                        flat[full_key] = step_tables[name][sa_key]
                        break
        result.append(flat)
    return result


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

    # Flatten to bootstrap-compatible format
    flat_db = _flatten_to_bootstrap_format(
        day_boundary_tables, stochastic, actions, config
    )
    flat_wd = _flatten_within_day(
        within_day_tables, stochastic, actions, config
    )

    # Save
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    db_path = _OUTPUT_DIR / "day_boundary.json"
    with db_path.open("w", encoding="utf-8") as f:
        json.dump(flat_db, f, indent=2)
    logger.info("Saved day_boundary.json with %d entries", len(flat_db))

    for step_idx, flat_table in enumerate(flat_wd):
        wd_path = _OUTPUT_DIR / f"within_day_{step_idx}.json"
        with wd_path.open("w", encoding="utf-8") as f:
            json.dump(flat_table, f, indent=2)
        logger.info(
            "Saved within_day_%d.json with %d entries", step_idx, len(flat_table)
        )

    logger.info("Done. Tables saved to %s", _OUTPUT_DIR)


if __name__ == "__main__":
    main()
