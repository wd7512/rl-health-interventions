"""Proxy for delayed effects via simplified MDP value iteration.

Solves the Bellman equation on a discretised dosage grid.
H(x, a) = expected next-state value after action a at dosage x.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ProxyValueFunction:
    """Computes the proxy for delayed effects on future rewards."""

    def __init__(
        self,
        decay: float = 0.95,
        gamma: float = 0.9,
        p_avail: float = 0.8,
        p_sed: float = 0.2,
        w: float = 0.5,
        grid_max: float = 20.0,
        grid_step: float = 0.5,
        treat_benefit: float = 2.0,
        burden_coef: float = 0.5,
    ) -> None:
        self.decay = decay
        self.gamma = gamma
        self.p_avail = p_avail
        self.p_sed = p_sed
        self.w = w
        self.treat_benefit = treat_benefit
        self.burden_coef = burden_coef
        self.grid = np.arange(0, grid_max + grid_step, grid_step)
        self.n_grid = len(self.grid)
        self._H_current = np.zeros((self.n_grid, 2))

    def _reward(self, x: float, a: int) -> float:
        if a == 0:
            return 0.0
        return self.treat_benefit * max(0.0, 1.0 - self.burden_coef * x)

    def _grid_index(self, x: float) -> int:
        return int(np.argmin(np.abs(self.grid - x)))

    def _transition_outcomes(self, x: float, a: int) -> list[tuple[float, float]]:
        if a == 1:
            return [(self.decay * x + 1.0, 1.0)]
        x_no = self.decay * x
        x_sed = self.decay * x + 1.0
        return [(x_no, 1.0 - self.p_sed), (x_sed, self.p_sed)]

    def _bellman_update(self, V: np.ndarray) -> np.ndarray:
        V_new = np.zeros((self.n_grid, 2))
        for i, x in enumerate(self.grid):
            H_vals = []
            for a in [0, 1]:
                H_a = 0.0
                for x_next, prob in self._transition_outcomes(x, a):
                    j = self._grid_index(x_next)
                    H_a += prob * (
                        self.p_avail * V[j, 1] + (1.0 - self.p_avail) * V[j, 0]
                    )
                H_vals.append(H_a)
            V_new[i, 0] = self._reward(x, 0) + self.gamma * H_vals[0]
            V_new[i, 1] = max(
                self._reward(x, 0) + self.gamma * H_vals[0],
                self._reward(x, 1) + self.gamma * H_vals[1],
            )
        return V_new

    def solve(self, tol: float = 1e-6, max_iter: int = 1000) -> dict:
        V = np.zeros((self.n_grid, 2))
        residual = 0.0
        n_iter = 0
        for n_iter in range(1, max_iter + 1):
            V_new = self._bellman_update(V)
            residual = float(np.max(np.abs(V_new - V)))
            V = V_new
            if residual < tol:
                break

        self._H_current = np.zeros((self.n_grid, 2))
        for i, x in enumerate(self.grid):
            for a in [0, 1]:
                H_a = 0.0
                for x_next, prob in self._transition_outcomes(x, a):
                    j = self._grid_index(x_next)
                    H_a += prob * (
                        self.p_avail * V[j, 1] + (1.0 - self.p_avail) * V[j, 0]
                    )
                self._H_current[i, a] = H_a

        return {"converged": residual < tol, "residual": residual, "iterations": n_iter}

    def H(self, x: float, a: int) -> float:
        return float(self._H_current[self._grid_index(x), a])

    def copy(self) -> ProxyValueFunction:
        pv = ProxyValueFunction(
            decay=self.decay,
            gamma=self.gamma,
            p_avail=self.p_avail,
            p_sed=self.p_sed,
            w=self.w,
            grid_max=float(self.grid[-1]),
            grid_step=float(self.grid[1] - self.grid[0]) if len(self.grid) > 1 else 0.5,
            treat_benefit=self.treat_benefit,
            burden_coef=self.burden_coef,
        )
        pv._H_current = self._H_current.copy()
        return pv
