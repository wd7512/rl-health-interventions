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


class ReinforceAgent(Agent):
    def __init__(
        self,
        actions: list[str] | None = None,
        lr: float = 1e-2,
        gamma: float = 0.99,
        hidden_dim: int | list[int] | None = None,
        state_dim: int = 64,
        seed: int = 42,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        self._rng = np.random.default_rng(seed)
        self._actions = actions or ["nudge", "idle"]
        self.lr = lr
        self.gamma = gamma
        self._state_dim = state_dim
        self._policy = MLP(
            input_dim=state_dim,
            output_dim=len(self._actions),
            hidden_dims=parse_hidden_dims(hidden_dim),
            seed=seed,
        )
        self._trajectory: list[tuple[np.ndarray, int, float]] = []

    def _encode(self, state) -> np.ndarray:
        return hash_state_vector(state, self._state_dim)

    def _action_probs(self, state_vec: np.ndarray) -> np.ndarray:
        return softmax(self._policy.predict(state_vec))

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
        self._trajectory.append((state_vec, action_idx, reward))

    @override
    def on_day_end(self) -> None:
        if not self._trajectory:
            return
        returns = np.zeros(len(self._trajectory), dtype=np.float64)
        running = 0.0
        for i in reversed(range(len(self._trajectory))):
            running = self._trajectory[i][2] + self.gamma * running
            returns[i] = running
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        for (state_vec, action_idx, _), advantage in zip(
            self._trajectory, returns, strict=False
        ):
            probs = self._action_probs(state_vec)
            grad_logits = probs
            grad_logits[action_idx] -= 1.0
            grad_logits *= advantage
            self._policy.backward_output_gradient(state_vec, grad_logits, lr=self.lr)
        self._trajectory.clear()
