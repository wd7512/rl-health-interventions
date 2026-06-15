from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.experiment import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="RL health interventions simulator")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to MDP YAML config"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="thompson_sampling",
        help="Agent name (default: thompson_sampling)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.csv",
        help="Output CSV path (default: results.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logging"
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    logger.info("Loaded config from %s", args.config)
    logger.info(
        "Episode: %d days x %d steps/day = %d timesteps",
        config.episode_days,
        config.steps_per_day,
        config.episode_days * config.steps_per_day,
    )

    agent = make_agent(args.agent)

    output_path = Path(args.output)
    df = run_episode(config, agent, output_csv=output_path, seed=args.seed)

    print("\n=== Episode complete ===")
    print(f"Total steps: {len(df)}")
    print(f"Total reward: {df['reward'].sum():.2f}")
    print(f"Mean reward per step: {df['reward'].mean():.4f}")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
