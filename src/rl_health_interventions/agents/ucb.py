from __future__ import annotations

import math

import numpy as np

from rl_health_interventions.agents._base import Agent
from rl_health_interventions.config.schemas import Action


class UCBAgent(Agent):
    """Upper Confidence Bound (UCB1) action selection.

    NOTE: This is a contextual bandit agent — state is accepted but not
    used in action selection. Q-values are updated globally, not per-state.
    For the MVP (Issue #101) this is correct. State-aware agents are
    planned for Phase 2.

    UCB1 selects actions by balancing exploitation (mean reward) with
    exploration (confidence interval width):

        a* = argmax_a [ Q(a) + c * sqrt(ln(N) / N_a) ]

    where Q(a) is the estimated value, N is total pulls, N_a is pulls
    for action a, and c is the exploration parameter.
    """

    def __init__(
        self,
        c: float = 2.0,
        seed: int = 42,
    ) -> None:
        if c <= 0.0:
            raise ValueError("c must be strictly positive.")
        self.c = c
        self._rng = np.random.default_rng(seed)
        self._actions = list(Action)
        self.q_values: dict[Action, float] = {action: 0.0 for action in Action}
        self.counts: dict[Action, int] = {action: 0 for action in Action}
        self._total_steps: int = 0

    def select_action(self, state) -> Action:
        # During initial exploration, pull each action once before applying UCB
        for action in self._actions:
            if self.counts[action] == 0:
                return action

        # UCB1: Q(a) + c * sqrt(ln(N) / N_a)
        total = self._total_steps
        ucb_values = {}
        for action in self._actions:
            n_a = self.counts[action]
            exploration = self.c * math.sqrt(math.log(total) / n_a)
            ucb_values[action] = self.q_values[action] + exploration

        max_ucb = max(ucb_values.values())
        best = [a for a, v in ucb_values.items() if v == max_ucb]
        idx = self._rng.integers(len(best))
        return best[idx]

    def update(self, state, action: Action, reward: float, next_state) -> None:
        self._total_steps += 1
        self.counts[action] += 1
        n = self.counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n


def register() -> None:
    from rl_health_interventions.agents import REGISTRY

    REGISTRY["ucb"] = UCBAgent
