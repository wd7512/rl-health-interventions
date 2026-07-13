from __future__ import annotations

import numpy as np

from rl_health_interventions.agents.deep_rl.dqn import DQNAgent
from rl_health_interventions.agents.deep_rl.ppo import PPOAgent
from rl_health_interventions.agents.deep_rl.q_learning import QLearningAgent
from rl_health_interventions.agents.deep_rl.reinforce import ReinforceAgent


def test_q_learning_updates_state_action_value(sed_and_act):
    sedentary_state, active_state = sed_and_act
    agent = QLearningAgent(actions=["nudge", "idle"], lr=0.5, gamma=0.0, epsilon=0.0)
    before = agent._values_for(sedentary_state)[0]
    agent.update(sedentary_state, "nudge", reward=1.0, next_state=active_state)
    after = agent._values_for(sedentary_state)[0]
    assert after > before


def test_dqn_updates_online_network(sedentary_state, active_state):
    agent = DQNAgent(
        actions=["nudge", "idle"],
        lr=0.05,
        gamma=0.0,
        epsilon=0.0,
        batch_size=1,
        buffer_size=10,
        target_update_freq=100,
        hidden_dim=8,
        state_dim=16,
        seed=7,
    )
    before = [w.copy() for w in agent._online.weights]
    agent.update(sedentary_state, "nudge", reward=1.0, next_state=active_state)
    after = agent._online.weights
    changed = any(
        not np.allclose(w_before, w_after)
        for w_before, w_after in zip(before, after, strict=False)
    )
    assert changed


def test_reinforce_on_day_end_updates_policy(sedentary_state):
    agent = ReinforceAgent(
        actions=["nudge", "idle"],
        lr=0.05,
        gamma=1.0,
        hidden_dim=8,
        state_dim=16,
        seed=3,
    )
    state_vec = agent._encode(sedentary_state)
    before = agent._action_probs(state_vec)[0]
    agent.update(sedentary_state, "nudge", reward=1.0, next_state=sedentary_state)
    agent.on_day_end()
    after = agent._action_probs(state_vec)[0]
    assert after != before


def test_ppo_on_day_end_clears_trajectory_and_updates_value(sedentary_state):
    agent = PPOAgent(
        actions=["nudge", "idle"],
        lr=0.05,
        gamma=1.0,
        gae_lambda=1.0,
        clip_eps=0.2,
        ppo_epochs=2,
        hidden_dim=8,
        state_dim=16,
        seed=5,
    )
    state_vec = agent._encode(sedentary_state)
    before = agent._value_estimate(state_vec)
    for _ in range(2):
        agent.update(sedentary_state, "nudge", reward=1.0, next_state=sedentary_state)
    agent.on_day_end()
    after = agent._value_estimate(state_vec)
    assert len(agent._trajectory) == 0
    assert after != before
