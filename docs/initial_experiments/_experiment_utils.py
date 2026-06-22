"""Shared utilities for initial experiments scripts."""
from __future__ import annotations

import numpy as np

from rl_health_interventions.agents import derive_agent_seed, make as make_agent
from rl_health_interventions.config.schemas import AgentConfig
from rl_health_interventions.experiment import run_episode


def run_agent(config, agent_cfg: AgentConfig, n_seeds: int) -> np.ndarray:
    """Run one agent variant over n_seeds. Returns per-step rewards (n_seeds, n_steps)."""
    rewards = []
    for seed in range(1, n_seeds + 1):
        kwargs = {k: v for k, v in agent_cfg.model_dump().items() if v is not None and k != "type"}
        kwargs["actions"] = config.action_names
        kwargs["seed"] = derive_agent_seed(seed, agent_index=0)
        agent = make_agent(agent_cfg.type, **kwargs)
        df = run_episode(config, agent, seed=seed)
        rewards.append(df["reward"].values)
    return np.array(rewards)
