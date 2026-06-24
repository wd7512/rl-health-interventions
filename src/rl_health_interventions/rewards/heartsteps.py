"""HeartSteps reward handler.

Computes the generative model reward:
    R = g(s)^T alpha_0 + f(s)^T alpha_1 + A * f(s)^T beta + noise
"""

from __future__ import annotations

import logging

import numpy as np

from rl_health_interventions.agents.heartsteps.features import (
    construct_heartsteps_features,
)
from rl_health_interventions.rewards._base import RewardHandler

logger = logging.getLogger(__name__)

_NIGHT_WINDOWS = {0, 9}


class HeartStepsReward(RewardHandler):
    """HeartSteps generative model reward handler.

    Computes rewards based on step data, feature construction, and a linear
    reward model with treatment effects.
    """

    def __init__(
        self,
        step_data: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        g_dim: int = 6,
        f_dim: int = 4,
        noise_variance: float = 1.0,
        p_avail: float = 0.85,
        p_sed: float = 0.2,
        n_windows: int = 10,
        seed: int = 42,
        config: object | None = None,
        **_kwargs: object,
    ) -> None:
        self._step_data = step_data
        self._alpha = alpha
        self._beta = beta
        self._g_dim = g_dim
        self._f_dim = f_dim
        self._noise_variance = noise_variance
        self._p_avail = p_avail
        self._p_sed = p_sed
        self._n_windows = n_windows
        self._rng = np.random.default_rng(seed)
        self._step_counter = 0
        self._daily_steps = 0.0

    def reward(self, state: str, action: str, step_idx: int) -> tuple[float, bool]:
        if step_idx == 0:
            self._daily_steps = 0.0

        t = self._step_counter
        day = t // self._n_windows
        window = t % self._n_windows
        time_slot = window // 2

        steps_flat = self._step_data.flatten()
        g, f = construct_heartsteps_features(
            steps_flat, t, self._daily_steps, time_slot, day
        )

        action_int = 1 if action == "suggest" else 0
        alpha_0 = self._alpha[: self._g_dim]
        alpha_1 = self._alpha[self._g_dim :]
        R_base = float(g @ alpha_0 + f @ alpha_1)
        f_beta = f[: len(self._beta)]
        R_treat = float(action_int * f_beta @ self._beta)
        epsilon = float(self._rng.normal(0, np.sqrt(self._noise_variance)))
        reward = R_base + R_treat + epsilon

        self._daily_steps += float(steps_flat[t])
        self._step_counter += 1

        return reward, False


def register() -> None:
    from rl_health_interventions.rewards import REGISTRY

    REGISTRY["heartsteps"] = HeartStepsReward
