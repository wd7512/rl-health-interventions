"""Proxy for delayed effects on future rewards via a simplified MDP.

Implements Section 5.4.2 of the HeartSteps V2 paper.

The simplified MDP models dosage transitions and computes the expected future
value V*(x, i) via value iteration on a discretised dosage grid. The proxy
H(x, a) = E[V*(x', i') | x, a] is the expected next-state value after taking
action a in dosage state x.

Reference:
    Liao et al. (2019). "Personalized HeartSteps." arXiv:1909.03539, Section 5.4.2.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ProxyValueFunction:
    """Computes the proxy for delayed effects on future rewards.

    The simplified MDP has state (dosage x, availability i). Availability is
    i.i.d. with probability p_avail. Dosage transitions follow:

        tau(x' | x, 1) = 1{x' = lambda * x + 1}
        tau(x' | x, 0) = p_sed * 1{x' = lambda * x + 1}
                        + (1 - p_sed) * 1{x' = lambda * x}

    The reward is r(x, a) = a * treat_benefit * max(0, 1 - burden_coef * x).
    At low dosage, sending produces a positive treatment effect. As dosage
    accumulates, the effect diminishes (habituation).

    Value iteration solves the Bellman equation on a discretised dosage grid.
    The proxy H(x, a) is the expected next-state value, and the delayed effect
    eta(x) = gamma * H(x, 0) - gamma * H(x, 1) captures the net future
    consequence of not sending vs sending a suggestion.
    """

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
        """Initialise the proxy value function.

        Args:
            decay: Dosage decay rate lambda.
            gamma: Discount factor.
            p_avail: Probability of being available.
            p_sed: Probability of anti-sedentary suggestion.
            w: Blending weight (H = (1-w)*H1 + w*H*).
            grid_max: Maximum dosage on the grid.
            grid_step: Grid step size.
            treat_benefit: Immediate reward benefit of taking action 1.
            burden_coef: Burden coefficient scaling the negative effect of dosage.
        """
        self.decay = decay
        self.gamma = gamma
        self.p_avail = p_avail
        self.p_sed = p_sed
        self.w = w
        self.treat_benefit = treat_benefit
        self.burden_coef = burden_coef

        self.grid = np.arange(0, grid_max + grid_step, grid_step)
        self.n_grid = len(self.grid)

        self._H1 = np.zeros((self.n_grid, 2))
        self._H_current = np.zeros((self.n_grid, 2))

        logger.debug(
            "ProxyValueFunction initialised: grid=%d points, gamma=%.2f, "
            "p_avail=%.2f, p_sed=%.2f, w=%.2f",
            self.n_grid,
            gamma,
            p_avail,
            p_sed,
            w,
        )

    def _reward(self, x: float, a: int) -> float:
        """Marginal reward r(x, a) depending only on dosage and action.

        Models the dosage-dependent treatment benefit:
            r(x, 1) = treat_benefit * max(0, 1 - burden_coef * x)
            r(x, 0) = 0

        At low dosage, sending a suggestion produces a positive treatment
        effect. As dosage accumulates, the effect diminishes (habituation).
        At high dosage (x > 1/burden_coef), the treatment effect is zero
        and sending is wasteful.

        Args:
            x: Current dosage.
            a: Binary action (0 or 1).

        Returns:
            Reward value.
        """
        if a == 0:
            return 0.0
        treatment_effect = self.treat_benefit * max(0.0, 1.0 - self.burden_coef * x)
        return treatment_effect

    def _grid_index(self, x: float) -> int:
        """Find nearest grid index for a dosage value.

        Args:
            x: Dosage value.

        Returns:
            Index into self.grid.
        """
        return int(np.argmin(np.abs(self.grid - x)))

    def _transition_outcomes(self, x: float, a: int) -> list[tuple[float, float]]:
        """Get possible next dosage values and their probabilities.

        Args:
            x: Current dosage.
            a: Binary action.

        Returns:
            List of (next_dosage, probability) pairs.
        """
        if a == 1:
            return [(self.decay * x + 1.0, 1.0)]
        else:
            x_no = self.decay * x
            x_sed = self.decay * x + 1.0
            return [(x_no, 1.0 - self.p_sed), (x_sed, self.p_sed)]

    def _bellman_update(self, V: np.ndarray) -> np.ndarray:
        """Apply one Bellman operator to the value function.

        Args:
            V: Current value function, shape (n_grid, 2).

        Returns:
            Updated value function.
        """
        V_new = np.zeros((self.n_grid, 2))

        for i, x in enumerate(self.grid):
            r0 = self._reward(x, 0)
            r1 = self._reward(x, 1)

            H_vals = []
            for a in [0, 1]:
                H_a = 0.0
                for x_next, prob in self._transition_outcomes(x, a):
                    j = self._grid_index(x_next)
                    H_a += prob * (
                        self.p_avail * V[j, 1] + (1.0 - self.p_avail) * V[j, 0]
                    )
                H_vals.append(H_a)

            V_new[i, 0] = r0 + self.gamma * H_vals[0]
            V_new[i, 1] = max(
                r0 + self.gamma * H_vals[0],
                r1 + self.gamma * H_vals[1],
            )

        return V_new

    def solve(self, tol: float = 1e-6, max_iter: int = 1000) -> dict:
        """Run value iteration on the dosage grid until convergence.

        Args:
            tol: Convergence tolerance (max Bellman residual).
            max_iter: Maximum number of iterations.

        Returns:
            Dict with keys 'converged' (bool), 'residual' (float),
            'iterations' (int).
        """
        V = np.zeros((self.n_grid, 2))

        for n_iter in range(1, max_iter + 1):
            V_new = self._bellman_update(V)
            residual = float(np.max(np.abs(V_new - V)))
            V = V_new

            if residual < tol:
                logger.debug(
                    "Value iteration converged in %d iterations, residual=%.2e",
                    n_iter,
                    residual,
                )
                break
        else:
            logger.warning(
                "Value iteration did not converge in %d iterations, residual=%.2e",
                max_iter,
                residual,
            )

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

        return {
            "converged": residual < tol,
            "residual": residual,
            "iterations": n_iter,
        }

    def H(self, x: float, a: int) -> float:
        """Get the proxy value H(x, a).

        Args:
            x: Dosage value.
            a: Binary action.

        Returns:
            Expected next-state value after action a at dosage x.
        """
        i = self._grid_index(x)
        return float(self._H_current[i, a])

    def eta(self, x: float) -> float:
        """Compute the delayed effect eta(x).

        eta(x) = gamma * H(x, 0) - gamma * H(x, 1)

        A positive value means not sending leads to better future outcomes.

        Args:
            x: Dosage value.

        Returns:
            Delayed effect value.
        """
        i = self._grid_index(x)
        return self.gamma * (self._H_current[i, 0] - self._H_current[i, 1])

    def update(self) -> None:
        """Weighted update blending H_1 (initial proxy) with current H*.

        H = (1 - w) * H_1 + w * H_current
        """
        self._H_current = (1.0 - self.w) * self._H1 + self.w * self._H_current
        logger.debug(
            "Proxy value updated: w=%.2f, H range=[%.4f, %.4f]",
            self.w,
            float(np.min(self._H_current)),
            float(np.max(self._H_current)),
        )
