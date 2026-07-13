from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import polars as pl

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.environment import Environment

logger = logging.getLogger(__name__)


class AgentLike(Protocol):
    def select_action(self, state) -> str: ...
    def update(self, state, action, reward, next_state) -> None: ...
    def on_day_end(self) -> None: ...


def run_episode(
    config: MDPConfig,
    agent: AgentLike,
    output_csv: Path | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Run a single episode and return results as a list of dicts.

    For better performance with large episodes, use run_episode_polars() which
    returns results as a Polars DataFrame.
    """
    if seed is None:
        seed = config.seed
    env = Environment(config, seed=seed)
    state = env.reset()
    records: list[dict] = []
    done = False
    declared_vars = set(config.state.variables.keys())
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state_factors = {
            k: v for k, v in state.factor_values.items() if k in declared_vars
        }
        records.append(
            {
                "step": state.global_step,
                "day": state.day,
                "step_of_day": state.step_of_day,
                **state_factors,
                "action": action,
                "reward": reward,
            }
        )
        agent.update(state, action, reward, next_state)
        if next_state.step_of_day == 0 and next_state.day > 0:
            agent.on_day_end()
        state = next_state

    if output_csv and records:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use Polars for efficient CSV writing
        import polars as pl

        df = pl.DataFrame(records)
        df.write_csv(output_path)
        logger.info("Wrote %d rows to %s", len(records), output_path)
    return records


def run_episode_polars(
    config: MDPConfig,
    agent: AgentLike,
    output_csv: Path | None = None,
    seed: int | None = None,
) -> pl.DataFrame:
    """Run a single episode and return results as a Polars DataFrame.

    This is more memory-efficient than run_episode() for large episodes
    as it avoids intermediate list[dict] storage.
    """
    import polars as pl

    if seed is None:
        seed = config.seed
    env = Environment(config, seed=seed)
    state = env.reset()
    done = False
    declared_vars = set(config.state.variables.keys())

    # Pre-allocate schema for Polars DataFrame
    schema = {
        "step": pl.Int64,
        "day": pl.Int64,
        "step_of_day": pl.Int64,
        "action": pl.Utf8,
        "reward": pl.Float64,
    }
    for var in declared_vars:
        schema[var] = pl.Utf8

    # Build records as list of tuples for efficient DataFrame construction
    records: list[tuple] = []
    fieldnames = list(schema.keys())

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state_factors = {
            k: v for k, v in state.factor_values.items() if k in declared_vars
        }
        record = (
            state.global_step,
            state.day,
            state.step_of_day,
            *(state_factors.get(var, "") for var in declared_vars),
            action,
            reward,
        )
        records.append(record)
        agent.update(state, action, reward, next_state)
        if next_state.step_of_day == 0 and next_state.day > 0:
            agent.on_day_end()
        state = next_state

    # Create DataFrame from records
    df = pl.DataFrame(records, schema=schema, orient="row")

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(output_path)
        logger.info("Wrote %d rows to %s", len(df), output_path)

    return df
