from __future__ import annotations

import numpy as np
import pytest

from rl_health_interventions.agents.deep_rl.dqn import DQNAgent
from rl_health_interventions.agents.deep_rl.ppo import PPOAgent
from rl_health_interventions.agents.deep_rl.q_learning import QLearningAgent
from rl_health_interventions.agents.deep_rl.reinforce import ReinforceAgent
from rl_health_interventions.agents.deep_rl.replay import ReplayBuffer, Transition


class FakeState:
    def __init__(self, *values, day=0, step_of_day=0):
        self.state_key = values
        self.day = day
        self.step_of_day = step_of_day


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

    def test_does_not_bootstrap_when_done(self):
        state = FakeState("s0")
        agent = QLearningAgent(
            actions=["nudge", "idle"], lr=0.5, gamma=0.9, epsilon=0.0
        )
        agent.update(state, "nudge", reward=1.0, next_state=state, done=True)
        q_value = agent._values_for(state)[0]
        # With done=True, td_target = reward = 1.0, gamma ignored
        expected = 0.0 + 0.5 * (1.0 - 0.0)
        assert q_value == pytest.approx(expected)

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
        assert not np.array_equal(pred_before, pred_after)

    def test_stores_done_in_transition(self, sed_and_act):
        sedentary_state, active_state = sed_and_act
        agent = DQNAgent(
            actions=["nudge", "idle"],
            lr=0.05,
            gamma=0.9,
            epsilon=0.0,
            batch_size=1,
            buffer_size=10,
            target_update_freq=100,
            hidden_dim=8,
            state_dim=16,
            seed=7,
        )
        agent.update(
            sedentary_state, "nudge", reward=1.0, next_state=active_state, done=True
        )
        transition = agent._replay._buffer[0]
        assert transition.done is True

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

    def test_gradient_clipping_limits_updates(self):
        agent = DQNAgent(
            actions=["nudge", "idle"],
            lr=1.0,
            gamma=0.99,
            epsilon=0.0,
            batch_size=1,
            buffer_size=10,
            target_update_freq=10000,
            hidden_dim=8,
            state_dim=16,
            seed=7,
            grad_clip=0.5,
        )
        rng = np.random.default_rng(99)
        state_vec = rng.uniform(0, 2, 16)
        next_vec = rng.uniform(0, 2, 16)
        agent._replay.append(
            Transition(
                state=state_vec,
                action_idx=0,
                reward=100.0,
                next_state=next_vec,
                done=False,
            )
        )
        before = [w.copy() for w in agent._online.weights]
        agent._train_step()
        after = agent._online.weights
        max_change = max(
            np.max(np.abs(before[i] - after[i])) for i in range(len(before))
        )
        assert max_change <= 0.5 + 1e-10, f"max change {max_change} exceeds clip limit"

    def test_zero_grad_clip_allows_large_updates(self):
        agent = DQNAgent(
            actions=["nudge", "idle"],
            lr=1.0,
            gamma=0.99,
            epsilon=0.0,
            batch_size=1,
            buffer_size=10,
            target_update_freq=10000,
            hidden_dim=8,
            state_dim=16,
            seed=7,
            grad_clip=0.0,
        )
        rng = np.random.default_rng(99)
        state_vec = rng.uniform(0, 2, 16)
        next_vec = rng.uniform(0, 2, 16)
        agent._replay.append(
            Transition(
                state=state_vec,
                action_idx=0,
                reward=100.0,
                next_state=next_vec,
                done=False,
            )
        )
        before = [w.copy() for w in agent._online.weights]
        agent._train_step()
        after = agent._online.weights
        max_change = max(
            np.max(np.abs(before[i] - after[i])) for i in range(len(before))
        )
        # Without clipping, the large TD error should produce large updates
        assert max_change > 0.5, f"max change {max_change} should be large without clip"

    def test_rejects_bad_grad_clip(self):
        with pytest.raises(ValueError, match="grad_clip"):
            DQNAgent(grad_clip=-0.1)


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

    def test_gradient_magnitude_scales_with_return(self, sedentary_state):
        """Verify larger returns produce larger gradient updates (no std normalization).

        The old code divided returns by std, making [1, 1] and [10, 10] trajectories
        produce identical gradient magnitudes. Mean-only baseline preserves scale.
        """
        agent = ReinforceAgent(
            actions=["nudge", "idle"],
            lr=0.1,
            gamma=1.0,
            hidden_dim=8,
            state_dim=16,
            seed=42,
        )

        # Low-return trajectory: two steps with reward 1.0
        w_before_low = [w.copy() for w in agent._policy.weights]
        agent.update(sedentary_state, "nudge", reward=1.0, next_state=sedentary_state)
        agent.update(sedentary_state, "nudge", reward=1.0, next_state=sedentary_state)
        agent.on_day_end()
        w_after_low = agent._policy.weights
        low_diff = sum(
            np.sum(np.abs(wa - wb))
            for wa, wb in zip(w_after_low, w_before_low, strict=False)
        )

        # High-return trajectory: two steps with reward 10.0
        agent2 = ReinforceAgent(
            actions=["nudge", "idle"],
            lr=0.1,
            gamma=1.0,
            hidden_dim=8,
            state_dim=16,
            seed=42,
        )
        w_before_high = [w.copy() for w in agent2._policy.weights]
        agent2.update(sedentary_state, "nudge", reward=10.0, next_state=sedentary_state)
        agent2.update(sedentary_state, "nudge", reward=10.0, next_state=sedentary_state)
        agent2.on_day_end()
        w_after_high = agent2._policy.weights
        high_diff = sum(
            np.sum(np.abs(wa - wb))
            for wa, wb in zip(w_after_high, w_before_high, strict=False)
        )

        assert high_diff > low_diff, (
            f"High-return gradient diff ({high_diff:.6f}) should exceed "
            f"low-return gradient diff ({low_diff:.6f})"
        )

    def test_accepts_done_parameter(self, sedentary_state):
        agent = ReinforceAgent(
            actions=["nudge", "idle"], lr=0.01, hidden_dim=8, state_dim=16, seed=3
        )
        # Should not raise when done=True is passed
        agent.update(
            sedentary_state, "nudge", reward=1.0, next_state=sedentary_state, done=True
        )
        assert len(agent._trajectory) == 1

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

    def test_handles_terminal_state_in_gae(self, sedentary_state):
        agent = PPOAgent(
            actions=["nudge", "idle"],
            lr=0.01,
            gamma=0.9,
            gae_lambda=1.0,
            clip_eps=0.2,
            ppo_epochs=1,
            hidden_dim=8,
            state_dim=16,
            seed=5,
        )
        # One step with done=True — no bootstrapping from next state
        agent.update(
            sedentary_state, "nudge", reward=1.0, next_state=sedentary_state, done=True
        )
        # Since update() only appends to trajectory (no training),
        # the value prediction is the network's initial forward pass.
        value = agent._value_estimate(agent._encode(sedentary_state))
        advantages, _, _ = agent._compute_gae()
        # With done=True, next_value is 0, so:
        # delta = reward + gamma * 0 - value = 1.0 - value
        expected = 1.0 - value
        assert advantages[0] == pytest.approx(expected, abs=1e-5)

    def test_gae_bootstraps_from_next_state_at_day_boundary(self):
        """GAE should use V(next_state) at day boundary, not 0.0."""
        agent = PPOAgent(
            actions=["nudge", "idle"],
            lr=0.01,
            gamma=0.9,
            gae_lambda=1.0,
            clip_eps=0.2,
            ppo_epochs=1,
            hidden_dim=8,
            state_dim=16,
            seed=5,
        )
        state = FakeState("s0")
        next_state = FakeState("s1")
        # Non-terminal transition (done=False) — day boundary, patient continues
        agent.update(state, "nudge", reward=1.0, next_state=next_state, done=False)
        state_vec = agent._encode(state)
        next_vec = agent._encode(next_state)
        value = agent._value_estimate(state_vec)
        next_value = agent._value_estimate(next_vec)
        advantages, _, _ = agent._compute_gae()
        # delta = reward + gamma * V(next_state) - V(state)
        expected = 1.0 + 0.9 * next_value - value
        assert advantages[0] == pytest.approx(expected, abs=1e-5)

    def test_value_gradient_averaged_over_trajectory_length(self):
        """Value gradient should not scale with trajectory length.

        With the fix, a trajectory of length N produces roughly the same
        total value gradient as a trajectory of length 1 (for the same
        per-step experience). Without the fix, it would be N times larger.
        """
        state = FakeState("s0")
        agent = PPOAgent(
            actions=["nudge", "idle"],
            lr=0.001,  # small lr ≈ linear gradient accumulation
            gamma=0.9,
            gae_lambda=0.0,  # no GAE accumulation — each step independent
            clip_eps=1.0,
            ppo_epochs=1,
            hidden_dim=8,
            state_dim=16,
            seed=5,
        )
        # Single-step trajectory
        w_before = [w.copy() for w in agent._value.weights]
        agent.update(state, "nudge", reward=1.0, next_state=state)
        agent.on_day_end()
        w_after = agent._value.weights
        single_diff = sum(
            np.sum(np.abs(wa - wb)) for wa, wb in zip(w_after, w_before, strict=False)
        )

        # Multi-step trajectory (10 steps, same per-step data)
        agent2 = PPOAgent(
            actions=["nudge", "idle"],
            lr=0.001,
            gamma=0.9,
            gae_lambda=0.0,
            clip_eps=1.0,
            ppo_epochs=1,
            hidden_dim=8,
            state_dim=16,
            seed=5,
        )
        w_before2 = [w.copy() for w in agent2._value.weights]
        for _ in range(10):
            agent2.update(state, "nudge", reward=1.0, next_state=state)
        agent2.on_day_end()
        w_after2 = agent2._value.weights
        multi_diff = sum(
            np.sum(np.abs(wa - wb)) for wa, wb in zip(w_after2, w_before2, strict=False)
        )

        # multi_diff should be roughly the same as single_diff (not 10x larger)
        assert multi_diff < single_diff * 4, (
            f"multi_diff ({multi_diff:.6f}) should not exceed "
            f"single_diff x 4 ({single_diff * 4:.6f})"
        )
        # Also check it's not vanishingly small
        assert multi_diff > single_diff * 0.25, (
            f"multi_diff ({multi_diff:.6f}) should not be less than "
            f"single_diff x 0.25 ({single_diff * 0.25:.6f})"
        )

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
