from __future__ import annotations

import csv
from pathlib import Path

from rl_health_interventions.agents.epsilon_greedy import EpsilonGreedyAgent
from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent
from rl_health_interventions.config.schemas import (
    ActivityLevel,
    Action,
    MDPConfig,
    TimeOfDay,
    TimeOfDayMask,
    TransitionMatrix,
)
from rl_health_interventions.experiment import run_episode


def _config(steps_per_day=5, episode_days=2) -> MDPConfig:
    return MDPConfig(
        activity_levels=[ActivityLevel.SEDENTARY, ActivityLevel.ACTIVE],
        actions=[Action.SEND, Action.DON_T_SEND],
        time_of_day=[
            TimeOfDay.MORNING,
            TimeOfDay.MIDDAY,
            TimeOfDay.AFTERNOON,
            TimeOfDay.EVENING,
            TimeOfDay.NIGHT,
        ],
        steps_per_day=steps_per_day,
        episode_days=episode_days,
        transition=TransitionMatrix(
            root={
                ActivityLevel.SEDENTARY: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.7,
                        ActivityLevel.ACTIVE: 0.3,
                    },
                    Action.DON_T_SEND: {
                        ActivityLevel.SEDENTARY: 0.9,
                        ActivityLevel.ACTIVE: 0.1,
                    },
                },
                ActivityLevel.ACTIVE: {
                    Action.SEND: {
                        ActivityLevel.SEDENTARY: 0.2,
                        ActivityLevel.ACTIVE: 0.8,
                    },
                    Action.DON_T_SEND: {
                        ActivityLevel.SEDENTARY: 0.4,
                        ActivityLevel.ACTIVE: 0.6,
                    },
                },
            }
        ),
        masks=TimeOfDayMask(
            root={
                t: {ActivityLevel.SEDENTARY: 0.0, ActivityLevel.ACTIVE: 0.0}
                for t in [
                    TimeOfDay.MORNING,
                    TimeOfDay.MIDDAY,
                    TimeOfDay.AFTERNOON,
                    TimeOfDay.EVENING,
                    TimeOfDay.NIGHT,
                ]
            }
        ),
        seed=42,
    )


def test_run_episode_returns_dataframe_with_correct_columns():
    import pandas as pd

    config = _config()
    agent = ThompsonSamplingAgent(seed=42)
    df = run_episode(config, agent, seed=42)
    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "step",
        "day",
        "step_of_day",
        "time_of_day",
        "state",
        "action",
        "reward",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 10  # 2 days x 5 steps/day


def test_run_episode_writes_csv(tmp_path: Path):
    config = _config()
    agent = EpsilonGreedyAgent(seed=42, epsilon=0.1)
    output_path = tmp_path / "results.csv"
    _df = run_episode(config, agent, output_csv=output_path, seed=42)
    assert output_path.exists()
    with output_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 10
    assert "step" in rows[0]
    assert "action" in rows[0]


def test_run_episode_reproducible_with_seed():
    config = _config()
    agent1 = ThompsonSamplingAgent(seed=42)
    agent2 = ThompsonSamplingAgent(seed=42)
    df1 = run_episode(config, agent1, seed=42)
    df2 = run_episode(config, agent2, seed=42)
    assert df1["action"].tolist() == df2["action"].tolist()
    assert df1["reward"].tolist() == df2["reward"].tolist()
