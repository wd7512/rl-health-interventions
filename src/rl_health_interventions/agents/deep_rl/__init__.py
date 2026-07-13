from __future__ import annotations

from rl_health_interventions.agents.deep_rl.dqn import DQNAgent
from rl_health_interventions.agents.deep_rl.ppo import PPOAgent
from rl_health_interventions.agents.deep_rl.q_learning import QLearningAgent
from rl_health_interventions.agents.deep_rl.reinforce import ReinforceAgent


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY.register("q_learning", QLearningAgent)
    REGISTRY.register("dqn", DQNAgent)
    REGISTRY.register("reinforce", ReinforceAgent)
    REGISTRY.register("ppo", PPOAgent)
