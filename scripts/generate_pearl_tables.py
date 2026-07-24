"""Generate PEARL 12-action transition tables using RandomTransitionSA.

Produces bootstrap-compatible JSON tables at tables/pearl_12action/.
Run once: uv run python scripts/generate_pearl_tables.py
"""
from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = (
    _REPOSITORY_ROOT
    / "docs"
    / "experimental_phases"
    / "pearl_random"
    / "configs"
    / "pearl_random.yaml"
)
_OUTPUT_DIR = _REPOSITORY_ROOT / "tables" / "pearl_12action"


def main() -> None:
    from rl_health_interventions.config.loader import load_config  # noqa: PLC0415
    from rl_health_interventions.transitions.random_sa import (  # noqa: PLC0415
        RandomTransitionSA,
    )

    logger.info("Loading config from %s", _CONFIG_PATH)
    config = load_config(_CONFIG_PATH)
    logger.info(
        "Config loaded: %d stochastic factors, %d actions, %d steps_per_day",
        len(
            [
                n
                for n, c in config.state.variables.items()
                if c.advanced is None
            ]
        ),
        len(config.actions),
        config.steps_per_day,
    )

    logger.info("Generating tables with seed=%d", config.seed)
    model = RandomTransitionSA(config, seed=config.seed)

    logger.info("Saving tables to %s", _OUTPUT_DIR)
    model.save_tables(_OUTPUT_DIR)
    logger.info("Done.")


if __name__ == "__main__":
    main()
