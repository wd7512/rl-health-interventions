"""Construct informative priors from training data.

Implements Section 5.5 of the HeartSteps V2 paper.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from rl_health_interventions.agents.heartsteps.features import (
    construct_heartsteps_features,
)

logger = logging.getLogger(__name__)


def _fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, p = X.shape
    if n <= p:
        return np.zeros(p), np.zeros(p), np.ones(p)
    try:
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank < p:
            beta = np.linalg.pinv(X) @ y
        residuals = y - X @ beta
        sigma2 = np.sum(residuals**2) / max(n - p, 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(sigma2 * np.abs(np.diag(XtX_inv)))
        z_stats = np.where(se > 1e-15, beta / se, 0.0)
        p_values = np.array(
            [
                2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
                for z in z_stats
            ]
        )
        p_values = np.clip(p_values, 0.0, 1.0)
        return beta, se, p_values
    except np.linalg.LinAlgError:
        return np.zeros(p), np.zeros(p), np.ones(p)


def _compute_daily_steps(
    steps: np.ndarray, n_windows: int, total_times: int
) -> np.ndarray:
    daily = np.zeros(total_times)
    for t in range(total_times):
        time_slot = t % n_windows // 2
        if time_slot == 0:
            daily[t] = steps[max(0, t - n_windows) : t].sum()
        else:
            daily[t] = daily[max(t - n_windows, 0)]
    return daily


def _extract_observations(
    step_data: np.ndarray,
    participant_idx: int,
    alpha: np.ndarray,
    beta: np.ndarray,
    g_dim: int,
    f_dim: int,
    noise_variance: float,
    p_avail: float,
    n_days: int,
    n_windows: int,
    pi_param: float = 0.3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    total_times = n_days * n_windows
    source = step_data[participant_idx]
    extra_idx = rng.integers(0, source.shape[0], size=max(0, n_days - source.shape[0]))
    extended = np.concatenate([source, source[extra_idx]], axis=0)[:n_days]
    steps = extended.flatten()
    daily_steps = _compute_daily_steps(steps, n_windows, total_times)
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for t in range(total_times):
        window = t % n_windows
        if window < 1 or window >= 9:
            continue
        available = rng.binomial(1, p_avail) == 1
        if not available:
            continue
        day = t // n_windows
        time_slot = window // 2
        g, f = construct_heartsteps_features(
            steps, t, float(daily_steps[t]), time_slot, day
        )
        action = int(rng.binomial(1, 0.3))
        alpha_0 = alpha[:g_dim]
        alpha_1 = alpha[g_dim:]
        reward = float(
            g @ alpha_0
            + f @ alpha_1
            + action * f[: len(beta)] @ beta
            + rng.normal(0, np.sqrt(noise_variance))
        )
        phi = np.concatenate([g, pi_param * f, (action - pi_param) * f])
        X_list.append(phi)
        y_list.append(reward)
    if not X_list:
        return np.zeros((0, g_dim + 2 * f_dim)), np.array([])
    return np.array(X_list), np.array(y_list)


def construct_prior(
    training_indices: list[int],
    step_data: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    g_dim: int = 6,
    f_dim: int = 4,
    n_days: int = 90,
    pi_param: float = 0.3,
    noise_variance: float = 1.0,
    p_avail: float = 0.85,
    alpha_level: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    total_params = g_dim + 2 * f_dim
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    per_part_betas: list[np.ndarray] = []
    for idx in training_indices:
        X_i, y_i = _extract_observations(
            step_data,
            idx,
            alpha,
            beta,
            g_dim,
            f_dim,
            noise_variance,
            p_avail,
            n_days,
            10,
            pi_param,
            rng,
        )
        all_X.append(X_i)
        all_y.append(y_i)
        if X_i.shape[0] > total_params:
            beta_i, _, _ = _fit_ols(X_i, y_i)
            per_part_betas.append(beta_i)
    if not all_X or all(X.shape[0] == 0 for X in all_X):
        return np.zeros(total_params), np.eye(total_params)
    X_pooled = np.vstack(all_X)
    y_pooled = np.concatenate(all_y)
    pooled_beta, pooled_se, pooled_p = _fit_ols(X_pooled, y_pooled)
    if len(per_part_betas) >= 2:
        cross_std = np.std(np.array(per_part_betas), axis=0, ddof=1)
    elif len(per_part_betas) == 1:
        cross_std = pooled_se.copy()
    else:
        cross_std = np.ones(total_params)
    prior_mean = np.zeros(total_params)
    prior_std = np.zeros(total_params)
    for k in range(total_params):
        if pooled_p[k] < alpha_level:
            prior_mean[k] = pooled_beta[k]
            prior_std[k] = cross_std[k]
        else:
            prior_mean[k] = 0.0
            prior_std[k] = cross_std[k] * 0.5
    prior_std = np.maximum(prior_std, 1e-8)
    return prior_mean, np.diag(prior_std**2)
