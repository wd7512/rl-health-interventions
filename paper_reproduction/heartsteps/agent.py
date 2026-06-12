"""Thompson Sampling agent with probability clipping.

Implements Section 5.3 of the HeartSteps V2 paper.

The agent selects actions using Thompson Sampling:
    1. Sample beta from posterior N(mu_d, Sigma_d).
    2. Compute action values Q(s, 0) and Q(s, 1).
    3. Compute softmax probability pi_t, clipped to [epsilon_1, epsilon_0].
    4. Sample A_t ~ Bernoulli(pi_t).

Reference:
    Liao et al. (2019). "Personalized HeartSteps." arXiv:1909.03539, Section 5.3.
"""

from __future__ import annotations

import logging

import numpy as np

from paper_reproduction.heartsteps.bayesian_regression import BayesianRewardModel
from paper_reproduction.heartsteps.dosage import DosageTracker
from paper_reproduction.heartsteps.proxy_value import ProxyValueFunction

logger = logging.getLogger(__name__)


class ThompsonSamplingAgent:
    """Thompson Sampling agent for the HeartSteps RL algorithm.

    Selects binary actions (suggest activity or not) using Thompson Sampling
    with probability clipping. Tracks dosage and updates the Bayesian reward
    model after each step.
    """

    def __init__(
        self,
        model: BayesianRewardModel,
        dosage_tracker: DosageTracker,
        proxy_value: ProxyValueFunction,
        tau: float = 1.0,
        epsilon_0: float = 0.2,
        epsilon_1: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            model: BayesianRewardModel for posterior inference.
            dosage_tracker: DosageTracker for cumulative exposure.
            proxy_value: ProxyValueFunction for delayed effects.
            tau: Softmax temperature. Default 1.0.
            epsilon_0: Upper clip for action probability. Default 0.2.
            epsilon_1: Lower clip for action probability. Default 0.1.
            rng: NumPy random generator. Created if None.
        """
        self.model = model
        self.dosage_tracker = dosage_tracker
        self.proxy_value = proxy_value
        self.tau = tau
        self.epsilon_0 = epsilon_0
        self.epsilon_1 = epsilon_1
        self._rng = rng if rng is not None else np.random.default_rng()

        logger.debug(
            "ThompsonSamplingAgent initialised: tau=%.2f, eps_0=%.2f, eps_1=%.2f",
            tau,
            epsilon_0,
            epsilon_1,
        )

    def select_action(
        self,
        g: np.ndarray,
        f: np.ndarray,
        pi: float,
        available: bool,
    ) -> tuple[int, float]:
        """Select an action using Thompson Sampling.

        Args:
            g: Baseline feature vector.
            f: Treatment effect feature vector.
            pi: Randomization probability (used in action-centering).
            available: Whether the participant is available (I_t).

        Returns:
            Tuple of (action, probability_of_action_1).
        """
        if not available:
            return 0, 0.0

        beta_sample = self.model.sample_beta(self._rng)

        alpha_0 = self.model.posterior_mean[: self.model.g_dim]
        alpha_1 = self.model.posterior_mean[
            self.model.g_dim : self.model.g_dim + self.model.f_dim
        ]

        x = self.dosage_tracker.value

        # Equation (2): action values
        Q0 = float(
            g @ alpha_0
            + 0.0 * (f @ alpha_1)
            + (0.0 - pi) * (f @ beta_sample)
            + self.proxy_value.gamma * self.proxy_value.H(x, 0)
        )
        Q1 = float(
            g @ alpha_0
            + pi * (f @ alpha_1)
            + (1.0 - pi) * (f @ beta_sample)
            + self.proxy_value.gamma * self.proxy_value.H(x, 1)
        )

        exp0 = np.exp(Q0 / self.tau)
        exp1 = np.exp(Q1 / self.tau)
        prob_raw = exp1 / (exp0 + exp1)

        prob = float(np.clip(prob_raw, self.epsilon_1, self.epsilon_0))
        action = int(self._rng.binomial(1, prob))

        return action, prob

    def step(
        self,
        g: np.ndarray,
        f: np.ndarray,
        action: int,
        pi: float,
        reward: float,
        available: bool = True,
    ) -> None:
        """Update model and dosage tracker after one decision time.

        Args:
            g: Baseline features for this step.
            f: Treatment features for this step.
            action: Action taken A_t.
            pi: Randomization probability.
            reward: Observed reward R_t.
            available: Whether participant was available.
        """
        self.model.update(g, f, action, pi, reward, available=available)

        treatment_delivered = available and (action == 1)
        self.dosage_tracker.update(
            treatment_delivered=treatment_delivered,
        )

        logger.debug(
            "Agent step: action=%d, dosage=%.4f, available=%s",
            action,
            self.dosage_tracker.value,
            available,
        )
