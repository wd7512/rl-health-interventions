from __future__ import annotations

import logging
from pathlib import Path

from rl_health_interventions.agents import derive_agent_seed, make as make_agent
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.episode import run_episode

logger = logging.getLogger(__name__)


def run_experiment(config_path: str | Path) -> dict[str, float]:
    """Run all agents defined in the config, return {agent_type: total_reward}."""
    config = load_config(config_path)
    results: dict[str, float] = {}
    for i, agent_cfg in enumerate(config.agents):
        kwargs = {
            k: v
            for k, v in agent_cfg.model_dump().items()
            if v is not None and k != "type"
        }
        kwargs["actions"] = config.actions
        kwargs["seed"] = derive_agent_seed(config.seed, agent_index=i)
        agent = make_agent(agent_cfg.type, **kwargs)
        records = run_episode(config, agent, seed=config.seed)
        key = agent_cfg.type
        suffix = 1
        while key in results:
            key = f"{agent_cfg.type}_{suffix}"
            suffix += 1
        results[key] = sum(r["reward"] for r in records)
    return results
