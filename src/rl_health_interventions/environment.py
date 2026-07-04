from __future__ import annotations

import logging

from rl_health_interventions import make_reward, make_transition
from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView

logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        self._config = config
        self._transition = make_transition(config, seed=seed)
        self._reward = make_reward(config)
        self._step_count = 0
        self._done = False
        self._current_state: StateView | None = None

    def reset(self) -> StateView:
        self._step_count = 0
        self._done = False
        self._current_state = StateView(
            factors=dict(self._config.initial_state),
            day=0,
            step_of_day=0,
            steps_per_day=self._config.steps_per_day,
        )
        logger.debug("Environment reset: %s", self._current_state)
        return self._current_state

    def step(self, action: str) -> tuple[StateView, float, bool]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._current_state is None:
            raise RuntimeError("Call reset() before step().")

        primary_var = next(iter(self._config.state.variables))
        current_val = getattr(self._current_state, primary_var)
        next_val = self._transition.transition(current_val, action)

        step_idx = self._step_count % self._config.steps_per_day
        reward, _ = self._reward.reward(self._current_state, action, step_idx)

        self._step_count += 1
        self._current_state = self._current_state.with_factors(
            **{primary_var: next_val}
        )
        self._current_state = self._current_state.with_advance()

        if self._step_count >= self._config.steps_per_day * self._config.episode_days:
            self._done = True

        logger.debug(
            "Step %d: action=%s, next=%s, reward=%.2f, done=%s",
            self._step_count,
            action,
            self._current_state,
            reward,
            self._done,
        )
        return self._current_state, reward, self._done
