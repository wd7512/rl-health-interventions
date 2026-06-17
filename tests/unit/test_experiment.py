from __future__ import annotations

import csv
from pathlib import Path

from rl_health_interventions.agents.epsilon_greedy import EpsilonGreedyAgent
from rl_health_interventions.agents.thompson_sampling import ThompsonSamplingAgent
from rl_health_interventions.experiment import run_episode


def test_run_episode_returns_dataframe_with_correct_columns(valid_config):
    import pandas as pd

    agent = ThompsonSamplingAgent(actions=["nudge", "idle"], seed=42)
    df = run_episode(valid_config, agent, seed=42)
    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "step",
        "day",
        "step_of_day",
        "state",
        "action",
        "reward",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 450


def test_run_episode_writes_csv(tmp_path: Path, valid_config):
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], seed=42, epsilon=0.1)
    output_path = tmp_path / "results.csv"
    _df = run_episode(valid_config, agent, output_csv=output_path, seed=42)
    assert output_path.exists()
    with output_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 450
    assert "step" in rows[0]
    assert "action" in rows[0]


def test_run_episode_reproducible_with_seed(valid_config):
    agent1 = ThompsonSamplingAgent(actions=["nudge", "idle"], seed=42)
    agent2 = ThompsonSamplingAgent(actions=["nudge", "idle"], seed=42)
    df1 = run_episode(valid_config, agent1, seed=42)
    df2 = run_episode(valid_config, agent2, seed=42)
    assert df1["action"].tolist() == df2["action"].tolist()
    assert df1["reward"].tolist() == df2["reward"].tolist()
