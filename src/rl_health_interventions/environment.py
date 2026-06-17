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
            activity=self._config.initial_state,
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

        next_activity = self._transition.transition(
            self._current_state.activity, action
        )

        step_idx = self._step_count % self._config.steps_per_day
        reward, _ = self._reward.reward(next_activity, action, step_idx)

        self._step_count += 1
        next_step_of_day = self._step_count % self._config.steps_per_day
        next_day = self._step_count // self._config.steps_per_day
        self._current_state = StateView(
            activity=next_activity,
            day=next_day,
            step_of_day=next_step_of_day,
            steps_per_day=self._config.steps_per_day,
        )

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
