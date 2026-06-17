from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

from rl_health_interventions.agents import make as make_agent
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
) -> pd.DataFrame:
    if seed is None:
        seed = config.seed
    env = Environment(config, seed=seed)
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
        kwargs["seed"] = (config.seed + i * 2654435761) % (2**31)
        agent = make_agent(agent_cfg.type, **kwargs)
        df = run_episode(config, agent, seed=config.seed)
        results[agent_cfg.type] = float(df["reward"].sum())
    return results
