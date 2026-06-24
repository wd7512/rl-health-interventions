"""HeartSteps V2 Thompson Sampling agent with action-centering."""

from __future__ import annotations

import logging

import numpy as np

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.agents.heartsteps.bayesian import BayesianRewardModel
from rl_health_interventions.agents.heartsteps.dosage import DosageTracker
from rl_health_interventions.agents.heartsteps.features import (
    construct_heartsteps_features,
)
from rl_health_interventions.agents.heartsteps.proxy import ProxyValueFunction
from rl_health_interventions.state import StateView

logger = logging.getLogger(__name__)

_NIGHT_WINDOWS = {0, 9}


class HeartStepsAgent(Agent):
    """HeartSteps V2 Thompson Sampling agent."""

    def __init__(
        self,
        step_data: np.ndarray,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        g_dim: int = 6,
        f_dim: int = 4,
        noise_variance: float = 1.0,
        pi_param: float = 0.3,
        tau: float = 1.0,
        epsilon_0: float = 0.2,
        epsilon_1: float = 0.1,
        dosage_decay: float = 0.95,
        proxy_gamma: float = 0.9,
        proxy_w: float = 0.5,
        proxy_p_avail: float = 0.85,
        proxy_p_sed: float = 0.2,
        proxy_grid_max: float = 20.0,
        proxy_grid_step: float = 0.5,
        treat_benefit: float = 2.0,
        burden_coef: float = 0.3,
        p_avail: float = 0.85,
        p_sed: float = 0.2,
        seed: int = 42,
        actions: list[str] | None = None,
        **_kwargs: object,
    ) -> None:
        self._step_data = step_data
        self._g_dim = g_dim
        self._f_dim = f_dim
        self._pi_param = pi_param
        self._tau = tau
        self._epsilon_0 = epsilon_0
        self._epsilon_1 = epsilon_1
        self._p_avail = p_avail
        self._p_sed = p_sed
        self._n_windows = 10
        self._rng = np.random.default_rng(seed)
        self._daily_steps = 0.0
        self._last_pi = pi_param
        self._last_available = True
        self._model = BayesianRewardModel(
            g_dim=g_dim,
            f_dim=f_dim,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            noise_variance=noise_variance,
        )
        self._dosage = DosageTracker(decay=dosage_decay)
        self._proxy = ProxyValueFunction(
            decay=dosage_decay,
            gamma=proxy_gamma,
            p_avail=proxy_p_avail,
            p_sed=proxy_p_sed,
            w=proxy_w,
            grid_max=proxy_grid_max,
            grid_step=proxy_grid_step,
            treat_benefit=treat_benefit,
            burden_coef=burden_coef,
        )
        self._proxy.solve()

    def select_action(self, state: StateView) -> str:
        day = state.day
        window = state.step_of_day
        t = day * self._n_windows + window
        time_slot = window // 2
        is_night = window in _NIGHT_WINDOWS
        available = False if is_night else self._rng.binomial(1, self._p_avail) == 1
        self._last_available = available
        if not available:
            self._last_pi = self._pi_param
            return "idle"
        steps_flat = self._step_data.flatten()
        g, f = construct_heartsteps_features(
            steps_flat, t, self._daily_steps, time_slot, day
        )
        beta_sample = self._model.sample_beta(self._rng)
        alpha_0 = self._model.posterior_mean[: self._g_dim]
        alpha_1 = self._model.posterior_mean[self._g_dim : self._g_dim + self._f_dim]
        x = self._dosage.value
        Q0 = float(
            g @ alpha_0
            + self._pi_param * (f @ alpha_1)
            + (0.0 - self._pi_param) * (f @ beta_sample)
            + self._proxy.gamma * self._proxy.H(x, 0)
        )
        Q1 = float(
            g @ alpha_0
            + self._pi_param * (f @ alpha_1)
            + (1.0 - self._pi_param) * (f @ beta_sample)
            + self._proxy.gamma * self._proxy.H(x, 1)
        )
        exp0 = np.exp(Q0 / self._tau)
        exp1 = np.exp(Q1 / self._tau)
        prob_raw = exp1 / (exp0 + exp1)
        prob = float(np.clip(prob_raw, self._epsilon_1, self._epsilon_0))
        action_int = int(self._rng.binomial(1, prob))
        self._last_pi = prob
        return "suggest" if action_int == 1 else "idle"

    def update(
        self,
        state: StateView,
        action: str,
        reward: float,
        next_state: StateView,
    ) -> None:
        day = state.day
        window = state.step_of_day
        t = day * self._n_windows + window
        time_slot = window // 2
        steps_flat = self._step_data.flatten()
        g, f = construct_heartsteps_features(
            steps_flat, t, self._daily_steps, time_slot, day
        )
        action_int = 1 if action == "suggest" else 0
        available = self._last_available
        self._model.update(
            g, f, action_int, self._pi_param, reward, available=available
        )
        self._dosage.update(
            treatment_delivered=available and action_int == 1,
        )
        self._daily_steps += float(steps_flat[t])
