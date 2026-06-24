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


def _fit_ridge(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, p = X.shape
    if n <= p:
        return np.zeros(p), np.ones(p), np.ones(p)
    try:
        XtX = X.T @ X + alpha * np.eye(p)
        beta = np.linalg.solve(XtX, X.T @ y)
        residuals = y - X @ beta
        sigma2 = np.sum(residuals**2) / max(n - p, 1)
        XtX_inv = np.linalg.inv(XtX)
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
        return np.zeros(p), np.ones(p), np.ones(p)


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


def construct_prior_from_steps(
    training_indices: list[int],
    step_data: np.ndarray,
    g_dim: int = 12,
    f_dim: int = 8,
    n_days: int = 42,
    n_windows: int = 10,
    pi_param: float = 0.3,
    p_avail: float = 0.85,
    weather_loader: object | None = None,
    participant_meta: list[dict] | None = None,
    alpha_level: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct prior from real step data without known generative model.

    Uses observed step counts as the outcome variable. The agent learns
    which contexts (features) are associated with higher step counts,
    and the prior encodes this aggregate knowledge.

    Args:
        training_indices: Indices of training participants in step_data.
        step_data: Array of shape (n_participants, n_days, n_windows).
        g_dim: Dimensionality of baseline features.
        f_dim: Dimensionality of treatment features.
        n_days: Number of days per episode.
        n_windows: Number of 30-minute windows per day.
        pi_param: Action probability for prior construction.
        p_avail: Availability probability.
        weather_loader: Optional WeatherLoader for weather features.
        participant_meta: Optional list of dicts with 'date' and 'region' keys.
        alpha_level: Significance level for prior inclusion.
        rng: Random generator.

    Returns:
        (prior_mean, prior_cov) tuple.
    """
    if rng is None:
        rng = np.random.default_rng()
    total_params = g_dim + 2 * f_dim
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    per_part_X: list[np.ndarray] = []
    per_part_y: list[np.ndarray] = []

    for idx in training_indices:
        source = step_data[idx]
        extra_idx = rng.integers(
            0, source.shape[0], size=max(0, n_days - source.shape[0])
        )
        extended = np.concatenate([source, source[extra_idx]], axis=0)[:n_days]
        steps = extended.flatten()
        total_times = n_days * n_windows
        daily_steps = _compute_daily_steps(steps, n_windows, total_times)

        weather_ctx: tuple[float, float] | None = None
        if weather_loader is not None and participant_meta is not None:
            from rl_health_interventions.data.weather import WeatherLoader

            if isinstance(weather_loader, WeatherLoader):
                meta = participant_meta[idx]
                weather_ctx = weather_loader.get_weather(meta["date"], meta["region"])

        X_list_p: list[np.ndarray] = []
        y_list_p: list[float] = []
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
                steps, t, float(daily_steps[t]), time_slot, day, weather_ctx
            )
            action = int(rng.binomial(1, pi_param))
            step_reward = float(steps[t])
            phi = np.concatenate([g, pi_param * f, (action - pi_param) * f])
            X_list_p.append(phi)
            y_list_p.append(step_reward)

        if X_list_p:
            X_arr = np.array(X_list_p)
            y_arr = np.array(y_list_p)
            all_X.append(X_arr)
            all_y.append(y_arr)
            per_part_X.append(X_arr)
            per_part_y.append(y_arr)

    if not all_X:
        return np.zeros(total_params), np.eye(total_params) * 10.0

    X_pooled = np.vstack(all_X)
    y_pooled = np.concatenate(all_y)

    X_mean = X_pooled.mean(axis=0)
    X_std = X_pooled.std(axis=0)
    X_std = np.maximum(X_std, 1e-8)
    X_normed = (X_pooled - X_mean) / X_std

    y_mean = y_pooled.mean()
    y_std = max(y_pooled.std(), 1e-8)
    y_normed = (y_pooled - y_mean) / y_std

    pooled_beta, pooled_se, pooled_p = _fit_ridge(X_normed, y_normed, alpha=1.0)

    prior_mean_raw = np.zeros(total_params)
    prior_std_raw = np.zeros(total_params)
    for k in range(total_params):
        if pooled_p[k] < alpha_level:
            prior_mean_raw[k] = pooled_beta[k] * (y_std / X_std[k])
            prior_std_raw[k] = pooled_se[k] * (y_std / X_std[k])
        else:
            prior_mean_raw[k] = 0.0
            prior_std_raw[k] = pooled_se[k] * (y_std / X_std[k]) * 0.5

    prior_std_raw = np.maximum(prior_std_raw, 1e-8)

    per_part_betas_normed: list[np.ndarray] = []
    for X_p, y_p in zip(per_part_X, per_part_y):
        if X_p.shape[0] > total_params:
            X_p_normed = (X_p - X_mean) / X_std
            y_p_normed = (y_p - y_mean) / y_std
            beta_i, _, _ = _fit_ridge(X_p_normed, y_p_normed, alpha=1.0)
            per_part_betas_normed.append(beta_i)

    if len(per_part_betas_normed) >= 2:
        cross_std = np.std(np.array(per_part_betas_normed), axis=0, ddof=1)
    elif len(per_part_betas_normed) == 1:
        cross_std = pooled_se.copy()
    else:
        cross_std = np.ones(total_params)

    prior_mean = np.clip(prior_mean_raw, -10.0, 10.0)
    prior_cov_diag = np.clip(cross_std * (y_std / np.maximum(X_std, 1e-8)), 1e-4, 10.0)

    return prior_mean, np.diag(prior_cov_diag**2)


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
            beta_i, _, _ = _fit_ridge(X_i, y_i)
            per_part_betas.append(beta_i)
    if not all_X or all(X.shape[0] == 0 for X in all_X):
        return np.zeros(total_params), np.eye(total_params)
    X_pooled = np.vstack(all_X)
    y_pooled = np.concatenate(all_y)
    pooled_beta, pooled_se, pooled_p = _fit_ridge(X_pooled, y_pooled)
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
