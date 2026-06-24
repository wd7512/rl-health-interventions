"""HeartSteps V2 agent package.

Exposes HeartStepsAgent, BayesianRewardModel, DosageTracker,
ProxyValueFunction, and construct_heartsteps_features.
"""

from rl_health_interventions.agents.heartsteps.agent import HeartStepsAgent
from rl_health_interventions.agents.heartsteps.bayesian import BayesianRewardModel
from rl_health_interventions.agents.heartsteps.dosage import DosageTracker
from rl_health_interventions.agents.heartsteps.proxy import ProxyValueFunction
from rl_health_interventions.agents.heartsteps.features import (
    construct_heartsteps_features,
)

__all__ = [
    "HeartStepsAgent",
    "BayesianRewardModel",
    "DosageTracker",
    "ProxyValueFunction",
    "construct_heartsteps_features",
]


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["heartsteps"] = HeartStepsAgent
