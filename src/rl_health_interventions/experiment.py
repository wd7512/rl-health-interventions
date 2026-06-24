from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from rl_health_interventions.agents import derive_agent_seed, make as make_agent
from rl_health_interventions.config.loader import load_config
from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.environment import Environment

logger = logging.getLogger(__name__)


class AgentLike(Protocol):
    def select_action(self, state) -> str: ...
    def update(self, state, action, reward, next_state) -> None: ...


def run_episode(
    config: MDPConfig,
    agent: AgentLike,
    output_csv: Path | None = None,
    seed: int | None = None,
    step_data: Any = None,
) -> pd.DataFrame:
    if seed is None:
        seed = config.seed
    env = Environment(config, seed=seed, step_data=step_data)
    state = env.reset()
    records: list[dict] = []
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        records.append(
            {
                "step": state.global_step,
                "day": state.day,
                "step_of_day": state.step_of_day,
                "state": state.activity,
                "action": action,
                "reward": reward,
            }
        )
        agent.update(state, action, reward, next_state)
        state = next_state

    df = pd.DataFrame.from_records(records)
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info("Wrote %d rows to %s", len(df), output_path)
    return df


def run_experiment(config_path: str | Path, step_data: Any = None) -> dict[str, float]:
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
        if agent_cfg.type == "heartsteps" and step_data is not None:
            kwargs["step_data"] = step_data
            kwargs["prior_mean"] = np.zeros(
                kwargs.get("g_dim", 6) + 2 * kwargs.get("f_dim", 4)
            )
            kwargs["prior_cov"] = np.eye(kwargs["g_dim"] + 2 * kwargs["f_dim"])
        agent = make_agent(agent_cfg.type, **kwargs)
        df = run_episode(config, agent, seed=config.seed, step_data=step_data)
        key = agent_cfg.type
        suffix = 1
        while key in results:
            key = f"{agent_cfg.type}_{suffix}"
            suffix += 1
        results[key] = float(df["reward"].sum())
    return results
