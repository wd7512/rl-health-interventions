from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rl_health_interventions.agents import derive_agent_seed
from rl_health_interventions.agents import make as make_agent
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.episode import run_episode

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="RL health interventions simulator")
    parser.add_argument(
        "--config",
        type=str,
        default="config/rule_based.yaml",
        help="Path to MDP YAML config (default: config/rule_based.yaml)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent name (default: first agent from config)",
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

    config = load_config(args.config)
    logger.info("Loaded config from %s", args.config)
    logger.info(
        "Episode: %d days x %d steps/day = %d timesteps",
        config.episode_days,
        config.steps_per_day,
        config.episode_days * config.steps_per_day,
    )

    agent_name = args.agent or (config.agents[0].type if config.agents else "random")
    env_seed = args.seed if args.seed is not None else config.seed
    agent_seed = derive_agent_seed(env_seed)

    agent_kwargs: dict[str, object] = {
        "seed": agent_seed,
        "actions": config.action_names,
    }
    agent_cfg = next((a for a in config.agents if a.type == agent_name), None)
    if agent_cfg is not None:
        agent_kwargs.update(agent_cfg.model_dump(exclude_unset=True, exclude={"type"}))
    agent = make_agent(agent_name, **agent_kwargs)

    output_path = Path(args.output)
    df = run_episode(config, agent, output_csv=output_path, seed=env_seed)

    logger.info("=== Episode complete ===")
    logger.info("Total steps: %d", len(df))
    total_reward = sum(r["reward"] for r in df)
    logger.info("Total reward: %.2f", total_reward)
    logger.info("Mean reward per step: %.4f", total_reward / len(df) if df else 0.0)
    logger.info("Results written to: %s", output_path)


if __name__ == "__main__":
    main()
