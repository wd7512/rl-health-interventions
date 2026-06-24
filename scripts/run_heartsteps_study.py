"""Run HeartSteps V2 simulation study."""

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    from rl_health_interventions.agents.heartsteps.bayesian import BayesianRewardModel
    from rl_health_interventions.agents.heartsteps.dosage import DosageTracker
    from rl_health_interventions.agents.heartsteps.proxy import ProxyValueFunction
    from rl_health_interventions.agents.heartsteps.features import construct_heartsteps_features
    from rl_health_interventions.data.nhanes import SyntheticNHANESGenerator
    from rl_health_interventions.simulation.cross_validation import create_folds
    from rl_health_interventions.simulation.prior import construct_prior

    logger.info("HeartSteps V2 Simulation Study")
    logger.info("=" * 50)

    g_dim, f_dim = 6, 4
    alpha = np.array([1.0, 0.5, 0.3, 0.1, 0.2, 0.1, 0.2, 0.15, 0.1, 0.05])
    beta = np.array([1.5, 2.5, 1.0, 0.5])
    noise_var, p_avail, p_sed = 1.0, 0.85, 0.2
    n_days, n_windows, pi_param = 90, 10, 0.3
    n_re_runs = 3

    rng = np.random.default_rng(42)
    step_data_raw = SyntheticNHANESGenerator(seed=42).generate(
        n_participants=30, n_days=7, n_windows=10
    )
    step_data = np.zeros((30, n_days, n_windows))
    for p in range(30):
        src = step_data_raw[p]
        extra_idx = rng.integers(0, src.shape[0], size=max(0, n_days - src.shape[0]))
        step_data[p] = np.concatenate([src, src[extra_idx]], axis=0)[:n_days]
    folds = create_folds(30, 3, rng=np.random.default_rng(42))
    all_improvements: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        logger.info("Fold %d", fold_idx + 1)
        prior_mean, prior_cov = construct_prior(
            train_idx.tolist(), step_data, alpha, beta, g_dim, f_dim,
            n_days, pi_param, noise_var, p_avail,
            rng=np.random.default_rng(int(rng.integers(0, 2**31))),
        )

        best_g, best_w, best_r = 0.0, 0.0, -float("inf")
        for gv in (0.0, 0.5, 0.9):
            for wv in (0.0, 0.5, 1.0):
                rews = []
                for p in train_idx[:3]:
                    for _ in range(2):
                        rews.append(_sim(step_data[int(p)], alpha, beta, g_dim, f_dim,
                            noise_var, p_avail, p_sed, n_days, n_windows, pi_param,
                            "prop", prior_mean, prior_cov, gv, wv,
                            int(rng.integers(0, 2**31))))
                mr = np.mean(rews) if rews else 0
                if mr > best_r:
                    best_r, best_g, best_w = mr, gv, wv
        logger.info("  Best gamma=%.2f w=%.2f", best_g, best_w)

        for p_idx in test_idx:
            prop_r, band_r = [], []
            for _ in range(n_re_runs):
                prop_r.append(_sim(step_data[int(p_idx)], alpha, beta, g_dim, f_dim,
                    noise_var, p_avail, p_sed, n_days, n_windows, pi_param,
                    "prop", prior_mean, prior_cov, best_g, best_w,
                    int(rng.integers(0, 2**31))))
                band_r.append(_sim(step_data[int(p_idx)], alpha, beta, g_dim, f_dim,
                    noise_var, p_avail, p_sed, n_days, n_windows, pi_param,
                    "bandit", prior_mean, prior_cov, 0.0, 0.0,
                    int(rng.integers(0, 2**31))))
            imp = float(np.mean(prop_r) - np.mean(band_r))
            all_improvements.append(imp)
            logger.info("  P%02d: prop=%.1f bandit=%.1f imp=%+.1f",
                int(p_idx), np.mean(prop_r), np.mean(band_r), imp)

    arr = np.array(all_improvements)
    n_imp = int(np.sum(arr > 0))
    logger.info("=" * 50)
    logger.info("RESULTS: %d/%d improved (%.1f%%)", n_imp, len(arr), 100*n_imp/len(arr))
    logger.info("Mean improvement: %+.2f", float(np.mean(arr)))
    logger.info("Median improvement: %+.2f", float(np.median(arr)))
    logger.info("Range: [%.2f, %.2f]", float(np.min(arr)), float(np.max(arr)))


def _sim(step_part, alpha, beta, g_dim, f_dim, noise_var, p_avail, p_sed,
         n_days, n_windows, pi_param, agent_type, prior_mean, prior_cov,
         proxy_gamma, proxy_w, seed):
    from rl_health_interventions.agents.heartsteps.bayesian import BayesianRewardModel
    from rl_health_interventions.agents.heartsteps.dosage import DosageTracker
    from rl_health_interventions.agents.heartsteps.proxy import ProxyValueFunction
    from rl_health_interventions.agents.heartsteps.features import construct_heartsteps_features

    rng = np.random.default_rng(seed)
    steps_flat = step_part.flatten()
    total_t = n_days * n_windows
    model = BayesianRewardModel(g_dim, f_dim, prior_mean.copy(), prior_cov.copy(), noise_var)
    dosage = DosageTracker(0.95)
    proxy = ProxyValueFunction(0.95, proxy_gamma, p_avail, p_sed, proxy_w, 20.0, 0.5, 2.0, 0.3)
    proxy.solve()
    daily_s, total_r = 0.0, 0.0
    for t in range(total_t):
        day, window = t // n_windows, t % n_windows
        ts = window // 2
        avail = False if window in {0, 9} else rng.binomial(1, p_avail) == 1
        g, f = construct_heartsteps_features(steps_flat, t, daily_s, ts, day)
        act_int, pi = 0, pi_param
        if avail:
            if agent_type == "prop":
                bs = model.sample_beta(rng)
                a0, a1 = model.posterior_mean[:g_dim], model.posterior_mean[g_dim:g_dim+f_dim]
                x = dosage.value
                Q0 = float(g @ a0 + pi_param*(f @ a1) + (0-pi_param)*(f @ bs) + proxy.gamma*proxy.H(x, 0))
                Q1 = float(g @ a0 + pi_param*(f @ a1) + (1-pi_param)*(f @ bs) + proxy.gamma*proxy.H(x, 1))
                p = float(np.clip(np.exp(Q1)/(np.exp(Q0)+np.exp(Q1)), 0.1, 0.2))
                act_int = int(rng.binomial(1, p))
                pi = p
            else:
                th = rng.multivariate_normal(model.posterior_mean, model.posterior_cov)
                Q0 = float(g @ th[:g_dim])
                Q1 = float(g @ th[:g_dim] + f @ th[g_dim:g_dim+f_dim])
                mq = max(Q0, Q1)
                p = float(np.clip(np.exp(Q1 - mq)/(np.exp(Q0 - mq)+np.exp(Q1 - mq)), 0.1, 0.2))
                act_int = int(rng.binomial(1, p))
        R = float(g @ alpha[:g_dim] + f @ alpha[g_dim:] + act_int * f[:len(beta)] @ beta + rng.normal(0, np.sqrt(noise_var)))
        if avail:
            if agent_type == "prop":
                model.update(g, f, act_int, pi_param, R, available=True)
            else:
                phi = np.concatenate([g, pi_param*f, (act_int-pi_param)*f])
                s2i = 1.0/noise_var
                Sinv = np.linalg.inv(model.posterior_cov)
                nc = np.linalg.inv(Sinv + s2i*np.outer(phi, phi))
                nm = nc @ (s2i*phi*R + Sinv @ model.posterior_mean)
                model._posterior_mean, model._posterior_cov = nm, nc
            dosage.update(treatment_delivered=act_int == 1)
        total_r += R
        daily_s += float(steps_flat[t])
    return total_r


if __name__ == "__main__":
    main()
