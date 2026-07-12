"""E2E benchmark: compare all contextual bandit agents on MVP configs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _shared import resolve_config

from rl_health_interventions.evaluation.runner import _positive_int, run_benchmark

logger = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
_CONFIG_REF_BASE = _CONFIGS_DIR.parent.parent

_MVP_CONFIGS = [
    _CONFIGS_DIR / "mvp.yaml",
    _CONFIGS_DIR / "mvp_masked.yaml",
    _CONFIGS_DIR / "mvp_extensions.yaml",
    _CONFIGS_DIR / "mvp_extensions_masked.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark contextual bandit agents")
    parser.add_argument("--seeds", type=_positive_int, default=50)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--confirm-overwrite", action="store_true")
    args = parser.parse_args()

    if args.json and args.output is None:
        parser.error("--json requires --output <dir>")

    n_seeds = args.seeds
    output_dir = Path(args.output) if args.output else None

    if args.all:
        for config_path in _MVP_CONFIGS:
            logger.info("\n=== Config: %s ===\n", config_path.name)
            run_benchmark(
                str(config_path),
                n_seeds,
                output_dir=output_dir,
                dump_json=args.json,
                confirm_overwrite=args.confirm_overwrite,
                config_ref_base=_CONFIG_REF_BASE,
            )
    else:
        run_benchmark(
            resolve_config(args.config),
            n_seeds,
            output_dir=output_dir,
            dump_json=args.json,
            confirm_overwrite=args.confirm_overwrite,
            config_ref_base=_CONFIG_REF_BASE,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
