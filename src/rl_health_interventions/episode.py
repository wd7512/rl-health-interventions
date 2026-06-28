from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Protocol

from rl_health_interventions.environment import Environment
from rl_health_interventions.config.schemas import MDPConfig

logger = logging.getLogger(__name__)


class AgentLike(Protocol):
    def select_action(self, state) -> str: ...
    def update(self, state, action, reward, next_state) -> None: ...


def run_episode(
    config: MDPConfig,
    agent: AgentLike,
    output_csv: Path | None = None,
    seed: int | None = None,
) -> list[dict]:
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

    if output_csv and records:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        logger.info("Wrote %d rows to %s", len(records), output_path)
    return records
