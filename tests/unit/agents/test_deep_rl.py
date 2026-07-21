from __future__ import annotations

import numpy as np
import pytest

from rl_health_interventions.agents.deep_rl.dqn import DQNAgent
from rl_health_interventions.agents.deep_rl.ppo import PPOAgent
from rl_health_interventions.agents.deep_rl.q_learning import QLearningAgent
from rl_health_interventions.agents.deep_rl.reinforce import ReinforceAgent
from rl_health_interventions.agents.deep_rl.replay import ReplayBuffer, Transition


class FakeState:
    def __init__(self, *values):
        self.state_key = values


@pytest.fixture
def sed_and_act():
    return FakeState("sedentary"), FakeState("active")


class TestQLearning:
    def test_updates_state_action_value(self, sed_and_act):
        sedentary_state, active_state = sed_and_act
        agent = QLearningAgent(
            actions=["nudge", "idle"], lr=0.5, gamma=0.0, epsilon=0.0
        )
        before = agent._values_for(sedentary_state)[0]
        agent.update(sedentary_state, "nudge", reward=1.0, next_state=active_state)
        after = agent._values_for(sedentary_state)[0]
        assert after > before

    def test_converges_on_deterministic_mdp(self):
        state = FakeState("s0")
        agent = QLearningAgent(
            actions=["nudge", "idle"], lr=0.5, gamma=0.9, epsilon=0.0
        )
        for _ in range(200):
            agent.update(state, "nudge", reward=1.0, next_state=state)
            agent.update(state, "idle", reward=0.0, next_state=state)
        q_nudge = agent._values_for(state)[0]
        q_idle = agent._values_for(state)[1]
        assert q_nudge > q_idle
        expected = 1.0 / (1.0 - 0.9)
        assert abs(q_nudge - expected) < 0.5

    def test_rejects_bad_lr(self):
        with pytest.raises(ValueError, match="lr"):
            QLearningAgent(lr=0)

    def test_rejects_bad_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            QLearningAgent(gamma=1.5)

    def test_rejects_bad_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            QLearningAgent(epsilon=-0.1)


class TestDQN:
    def test_updates_online_network(self, sed_and_act):
        sedentary_state, active_state = sed_and_act
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

    def test_loss_decreases_on_fixed_data(self, sed_and_act):
        sedentary_state, active_state = sed_and_act
        agent = DQNAgent(
            actions=["nudge", "idle"],
            lr=0.05,
            gamma=0.9,
            epsilon=0.0,
            batch_size=4,
            buffer_size=20,
            target_update_freq=10000,
            hidden_dim=8,
            state_dim=16,
            seed=7,
        )
        for _ in range(20):
            agent.update(sedentary_state, "nudge", reward=1.0, next_state=active_state)
        pred_before = agent._online.predict(agent._encode(sedentary_state)).copy()
        for _ in range(100):
            agent._train_step()
        pred_after = agent._online.predict(agent._encode(sedentary_state))
        assert not np.allclose(pred_before, pred_after)

    def test_target_sync(self, sed_and_act):
        sedentary_state, active_state = sed_and_act
        agent = DQNAgent(
            actions=["nudge", "idle"],
            lr=0.05,
            gamma=0.0,
            epsilon=0.0,
            batch_size=1,
            buffer_size=10,
            target_update_freq=2,
            hidden_dim=8,
            state_dim=16,
            seed=7,
        )
        target_before = [w.copy() for w in agent._target.weights]
        for _ in range(4):
            agent.update(sedentary_state, "nudge", reward=1.0, next_state=active_state)
        target_changed = any(
            not np.allclose(tb, tw)
            for tb, tw in zip(target_before, agent._target.weights, strict=False)
        )
        assert target_changed

    def test_rejects_bad_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            DQNAgent(batch_size=0)

    def test_rejects_bad_buffer_size(self):
        with pytest.raises(ValueError, match="buffer_size"):
            DQNAgent(buffer_size=0)

    def test_rejects_bad_target_update_freq(self):
        with pytest.raises(ValueError, match="target_update_freq"):
            DQNAgent(target_update_freq=0)


class TestReinforce:
    def test_on_day_end_updates_policy(self, sedentary_state):
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

    def test_increases_probability_after_positive_reward(self, sedentary_state):
        agent = ReinforceAgent(
            actions=["nudge", "idle"],
            lr=0.1,
            gamma=1.0,
            hidden_dim=8,
            state_dim=16,
            seed=3,
        )
        state_vec = agent._encode(sedentary_state)
        probs_before = agent._action_probs(state_vec).copy()
        for _ in range(20):
            agent.update(
                sedentary_state, "nudge", reward=1.0, next_state=sedentary_state
            )
            agent.on_day_end()
        probs_after = agent._action_probs(state_vec)
        assert probs_after[0] > probs_before[0]

    def test_clears_trajectory(self, sedentary_state):
        agent = ReinforceAgent(
            actions=["nudge", "idle"], lr=0.01, hidden_dim=8, state_dim=16, seed=3
        )
        agent.update(sedentary_state, "nudge", reward=1.0, next_state=sedentary_state)
        assert len(agent._trajectory) == 1
        agent.on_day_end()
        assert len(agent._trajectory) == 0

    def test_noop_on_empty_trajectory(self):
        agent = ReinforceAgent(
            actions=["nudge", "idle"], lr=0.01, hidden_dim=8, state_dim=16, seed=3
        )
        agent.on_day_end()

    def test_rejects_bad_lr(self):
        with pytest.raises(ValueError, match="lr"):
            ReinforceAgent(lr=0)

    def test_rejects_bad_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            ReinforceAgent(gamma=0)


class TestPPO:
    def test_on_day_end_clears_trajectory_and_updates_value(self, sedentary_state):
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
            agent.update(
                sedentary_state, "nudge", reward=1.0, next_state=sedentary_state
            )
        agent.on_day_end()
        after = agent._value_estimate(state_vec)
        assert len(agent._trajectory) == 0
        assert after != before

    def test_noop_on_empty_trajectory(self):
        agent = PPOAgent(
            actions=["nudge", "idle"],
            lr=0.01,
            clip_eps=0.2,
            ppo_epochs=1,
            hidden_dim=8,
            state_dim=16,
            seed=5,
        )
        agent.on_day_end()

    def test_separate_policy_and_value_networks(self):
        agent = PPOAgent(
            actions=["nudge", "idle"],
            policy_hidden_dim=[16],
            value_hidden_dim=[32],
            state_dim=16,
            seed=5,
        )
        assert agent._policy.weights[0].shape == (16, 16)
        assert agent._value.weights[0].shape == (16, 32)

    def test_rejects_bad_clip_eps(self):
        with pytest.raises(ValueError, match="clip_eps"):
            PPOAgent(clip_eps=0)

    def test_rejects_bad_ppo_epochs(self):
        with pytest.raises(ValueError, match="ppo_epochs"):
            PPOAgent(ppo_epochs=0)

    def test_rejects_bad_gae_lambda(self):
        with pytest.raises(ValueError, match="gae_lambda"):
            PPOAgent(gae_lambda=1.5)


class TestReplayBuffer:
    def test_append_and_len(self):
        buf = ReplayBuffer(capacity=10)
        t = Transition(
            state=np.zeros(4), action_idx=0, reward=1.0, next_state=np.zeros(4)
        )
        buf.append(t)
        assert len(buf) == 1

    def test_sample_without_replacement(self):
        buf = ReplayBuffer(capacity=10)
        rng = np.random.default_rng(0)
        for i in range(5):
            buf.append(
                Transition(
                    state=np.full(4, float(i)),
                    action_idx=0,
                    reward=float(i),
                    next_state=np.zeros(4),
                )
            )
        sample = buf.sample(3, rng)
        states = [tuple(t.state.tolist()) for t in sample]
        assert len(set(states)) == 3

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.append(
                Transition(
                    state=np.full(4, float(i)),
                    action_idx=0,
                    reward=float(i),
                    next_state=np.zeros(4),
                )
            )
        assert len(buf) == 3

    def test_sample_too_large_raises(self):
        buf = ReplayBuffer(capacity=10)
        buf.append(
            Transition(
                state=np.zeros(4),
                action_idx=0,
                reward=0.0,
                next_state=np.zeros(4),
            )
        )
        with pytest.raises(ValueError, match="cannot exceed"):
            buf.sample(5, np.random.default_rng(0))

    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity"):
            ReplayBuffer(capacity=0)

    def test_zero_batch_size_raises(self):
        buf = ReplayBuffer(capacity=10)
        with pytest.raises(ValueError, match="batch_size"):
            buf.sample(0, np.random.default_rng(0))
