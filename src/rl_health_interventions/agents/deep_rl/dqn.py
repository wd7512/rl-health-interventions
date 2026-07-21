from __future__ import annotations

import numpy as np
from typing_extensions import override

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.agents.deep_rl._base import (
    MLP,
    hash_state_vector,
    parse_hidden_dims,
    validate_gamma,
    validate_lr,
    validate_positive_int,
    validate_unit_interval,
)
from rl_health_interventions.agents.deep_rl.replay import ReplayBuffer, Transition


class DQNAgent(Agent):
    def __init__(
        self,
        actions: list[str] | None = None,
        lr: float = 1e-2,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_start: float | None = None,
        epsilon_min: float = 0.01,
        decay_steps: int = 1000,
        batch_size: int = 32,
        buffer_size: int = 1000,
        target_update_freq: int = 50,
        hidden_dim: int | list[int] | None = None,
        state_dim: int = 64,
        seed: int = 42,
    ) -> None:
        validate_lr(lr)
        validate_gamma(gamma)
        validate_unit_interval(epsilon, "epsilon")
        validate_unit_interval(epsilon_min, "epsilon_min")
        validate_positive_int(decay_steps, "decay_steps")
        validate_positive_int(batch_size, "batch_size")
        validate_positive_int(buffer_size, "buffer_size")
        validate_positive_int(target_update_freq, "target_update_freq")
        self._rng = np.random.default_rng(seed)
        self._actions = actions or ["nudge", "idle"]
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon if epsilon_start is None else epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_steps = decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._state_dim = state_dim
        hidden_dims = parse_hidden_dims(hidden_dim)
        self._online = MLP(
            input_dim=state_dim,
            output_dim=len(self._actions),
            hidden_dims=hidden_dims,
            seed=seed,
        )
        self._target = self._online.copy()
        self._replay = ReplayBuffer(capacity=buffer_size)
        self._steps = 0

    def _encode(self, state) -> np.ndarray:
        return hash_state_vector(state, self._state_dim)

    def _epsilon(self) -> float:
        frac = min(self._steps / self.decay_steps, 1.0)
        return self.epsilon_start - (self.epsilon_start - self.epsilon_min) * frac

    @override
    def select_action(self, state) -> str:
        if self._rng.random() < self._epsilon():
            return self._actions[int(self._rng.integers(len(self._actions)))]
        q_values = self._online.predict(self._encode(state))
        best = np.flatnonzero(q_values == np.max(q_values))
        idx = int(self._rng.choice(best))
        return self._actions[idx]

    def _train_step(self) -> None:
        if len(self._replay) < self.batch_size:
            return
        batch = self._replay.sample(self.batch_size, self._rng)
        total_grad_w, total_grad_b = self._accumulate_batch_gradients(batch)
        scale = self.lr / self.batch_size
        for idx in range(len(self._online.weights)):
            self._online.weights[idx] -= scale * total_grad_w[idx]
            self._online.biases[idx] -= scale * total_grad_b[idx]

    def _accumulate_batch_gradients(self, batch):
        n_actions = len(self._actions)
        total_grad_w = [np.zeros_like(w) for w in self._online.weights]
        total_grad_b = [np.zeros_like(b) for b in self._online.biases]
        for transition in batch:
            target_next = self._target.predict(transition.next_state)
            td_target = transition.reward + self.gamma * float(np.max(target_next))
            pred = self._online.predict(transition.state)
            grad_output = np.zeros(n_actions, dtype=np.float64)
            grad_output[transition.action_idx] = 2.0 * (
                pred[transition.action_idx] - td_target
            )
            grads_w, grads_b = self._online.compute_gradients(
                transition.state, grad_output
            )
            for idx in range(len(self._online.weights)):
                total_grad_w[idx] += grads_w[idx]
                total_grad_b[idx] += grads_b[idx]
        return total_grad_w, total_grad_b

    @override
    def update(self, state, action: str, reward: float, next_state) -> None:
        action_idx = self._actions.index(action)
        self._replay.append(
            Transition(
                state=self._encode(state),
                action_idx=action_idx,
                reward=reward,
                next_state=self._encode(next_state),
            )
        )
        self._steps += 1
        self._train_step()
        if self._steps % self.target_update_freq == 0:
            self._target.sync_from(self._online)
