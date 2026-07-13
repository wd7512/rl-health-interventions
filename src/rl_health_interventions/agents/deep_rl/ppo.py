from __future__ import annotations

import numpy as np
from typing_extensions import override

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.agents.deep_rl._base import (
    MLP,
    hash_state_vector,
    parse_hidden_dims,
    softmax,
)


class PPOAgent(Agent):
    def __init__(  # noqa: C901
        self,
        actions: list[str] | None = None,
        lr: float = 1e-2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        hidden_dim: int | list[int] | None = None,
        policy_hidden_dim: int | list[int] | None = None,
        value_hidden_dim: int | list[int] | None = None,
        state_dim: int = 64,
        seed: int = 42,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        if not (0.0 <= gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in [0, 1]")
        if clip_eps <= 0:
            raise ValueError("clip_eps must be > 0")
        if ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be > 0")
        self._rng = np.random.default_rng(seed)
        self._actions = actions or ["nudge", "idle"]
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self._state_dim = state_dim
        policy_dims = parse_hidden_dims(policy_hidden_dim or hidden_dim)
        value_dims = parse_hidden_dims(value_hidden_dim or hidden_dim)
        self._policy = MLP(
            input_dim=state_dim,
            output_dim=len(self._actions),
            hidden_dims=policy_dims,
            seed=seed,
        )
        self._value = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=value_dims,
            seed=seed + 1,
        )
        self._trajectory: list[tuple[np.ndarray, int, float, float, float]] = []

    def _encode(self, state) -> np.ndarray:
        return hash_state_vector(state, self._state_dim)

    def _action_probs(self, state_vec: np.ndarray) -> np.ndarray:
        return softmax(self._policy.predict(state_vec))

    def _value_estimate(self, state_vec: np.ndarray) -> float:
        return float(self._value.predict(state_vec)[0])

    @override
    def select_action(self, state) -> str:
        state_vec = self._encode(state)
        probs = self._action_probs(state_vec)
        idx = int(self._rng.choice(len(self._actions), p=probs))
        return self._actions[idx]

    @override
    def update(self, state, action: str, reward: float, next_state) -> None:
        state_vec = self._encode(state)
        action_idx = self._actions.index(action)
        probs = self._action_probs(state_vec)
        log_prob = float(np.log(probs[action_idx] + 1e-12))
        value = self._value_estimate(state_vec)
        self._trajectory.append((state_vec, action_idx, reward, value, log_prob))

    @override
    def on_day_end(self) -> None:  # noqa: C901, PLR0912
        if not self._trajectory:
            return
        n = len(self._trajectory)
        rewards = np.array([t[2] for t in self._trajectory], dtype=np.float64)
        values = np.array([t[3] for t in self._trajectory], dtype=np.float64)
        old_log_probs = np.array([t[4] for t in self._trajectory], dtype=np.float64)
        advantages = np.zeros(n, dtype=np.float64)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            for idx, (state_vec, action_idx, _, _, _) in enumerate(self._trajectory):
                probs = self._action_probs(state_vec)
                log_prob = float(np.log(probs[action_idx] + 1e-12))
                ratio = float(np.exp(log_prob - old_log_probs[idx]))
                advantage = float(advantages[idx])
                if (advantage >= 0.0 and ratio > (1.0 + self.clip_eps)) or (
                    advantage < 0.0 and ratio < (1.0 - self.clip_eps)
                ):
                    policy_coef = 0.0
                else:
                    policy_coef = -advantage * ratio
                grad_logits = -policy_coef * probs
                grad_logits[action_idx] += policy_coef
                self._policy.backward_output_gradient(
                    state_vec, grad_logits, lr=self.lr
                )

                value_pred = self._value_estimate(state_vec)
                value_grad = np.array(
                    [2.0 * (value_pred - returns[idx])], dtype=np.float64
                )
                self._value.backward_output_gradient(state_vec, value_grad, lr=self.lr)
        self._trajectory.clear()
