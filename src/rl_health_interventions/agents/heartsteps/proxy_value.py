from __future__ import annotations

import numpy as np


class ProxyValue:
    """Proxy for delayed effects using a simplified MDP over dosage.

    Maintains a value table H(dosage_bin, action) and updates it via
    Bellman backups with exponential moving average.
    """

    def __init__(
        self,
        actions: list[str],
        reference_action: str,
        dosage_grid_size: int = 20,
        gamma: float = 0.99,
        w: float = 0.5,
        lambda_dosage: float = 0.95,
        max_dosage: float = 5.0,
        suggestion_actions: list[str] | None = None,
    ) -> None:
        self._actions = list(actions)
        self._reference_action = reference_action
        self._gamma = gamma
        self._w = w
        self._lambda_dosage = lambda_dosage
        self._grid_size = dosage_grid_size
        self._max_dosage = max_dosage
        self._dosage_step = max_dosage / dosage_grid_size

        if suggestion_actions is not None:
            self._is_suggestion = {a: a in suggestion_actions for a in self._actions}
        else:
            self._is_suggestion = {a: a != reference_action for a in self._actions}

        self._H = np.zeros((dosage_grid_size, len(self._actions)))
        self._action_idx = {a: i for i, a in enumerate(self._actions)}

    def _discretize_dosage(self, dosage: float) -> int:
        idx = int(dosage / self._dosage_step)
        return min(max(idx, 0), self._grid_size - 1)

    def _next_dosage_idx(self, dosage_idx: int, action: str) -> int:
        dosage = dosage_idx * self._dosage_step
        if self._is_suggestion[action]:
            next_dosage = self._lambda_dosage * dosage + 1.0
        else:
            next_dosage = self._lambda_dosage * dosage
        return self._discretize_dosage(next_dosage)

    def update(
        self,
        reward_means: dict[str, float],
    ) -> None:
        """Bellman backup with exponential moving average."""
        h_new = np.zeros_like(self._H)
        for d_idx in range(self._grid_size):
            for a_name in self._actions:
                a_idx = self._action_idx[a_name]
                next_d = self._next_dosage_idx(d_idx, a_name)
                r = reward_means.get(a_name, 0.0)
                h_new[d_idx, a_idx] = r + self._gamma * float(np.max(self._H[next_d]))
        self._H = (1.0 - self._w) * self._H + self._w * h_new

    def get_eta(self, action: str, dosage: float) -> float:
        """Returns gamma * H(x, a_ref) - gamma * H(x, a)."""
        d_idx = self._discretize_dosage(dosage)
        a_idx = self._action_idx[action]
        ref_idx = self._action_idx[self._reference_action]
        return self._gamma * (self._H[d_idx, ref_idx] - self._H[d_idx, a_idx])

    @property
    def value_table(self) -> np.ndarray:
        return self._H.copy()

    @property
    def action_idx(self) -> dict[str, int]:
        return self._action_idx
