from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

from rl_health_interventions.config.schemas import Action, MDPConfig
from rl_health_interventions.environment import Environment

logger = logging.getLogger(__name__)


class AgentLike(Protocol):
    def select_action(self, state) -> Action: ...
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
                "time_of_day": state.time_of_day.value,
                "state": state.activity.value,
                "action": action.value,
                "reward": reward,
            }
        )
        agent.update(state, action, reward, next_state)
        state = next_state

    df = pd.DataFrame.from_records(records)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        logger.info("Wrote %d rows to %s", len(df), output_csv)
    return df
