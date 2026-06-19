from __future__ import annotations

import dataclasses

import numpy as np

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.state import StateView
from rl_health_interventions.transitions._base import TransitionModel


class RuleBasedTransition(TransitionModel):
    def __init__(self, config: MDPConfig, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._cache: dict[tuple[str, str], tuple[list[str], np.ndarray]] = {}
        self._state_dynamics = config.transition_model.state_dynamics
        if config.transition_model.transition_probabilities is not None:
            self._build_cache(config)

    def _build_cache(self, config: MDPConfig) -> None:
        prob_config = config.transition_model.transition_probabilities
        if prob_config is None:
            raise ValueError("rule_based transition requires transition_probabilities")
        for state, actions in prob_config.root.items():
            for action, probs in actions.items():
                targets = list(probs.keys())
                prob_values = np.array(list(probs.values()), dtype=np.float64)
                prob_values /= prob_values.sum()
                self._cache[(state, action)] = (targets, prob_values)

    def _sample_activity(self, state: StateView, action: str) -> str:
        targets, probs = self._cache[(state.activity, action)]
        idx = self._rng.choice(len(targets), p=probs)
        return targets[idx]

    def _compute_time_of_day(self, step_of_day: int, steps_per_day: int) -> int:
        return int(step_of_day * 24 / steps_per_day)

    def _compute_day_of_week(self, day: int) -> int:
        return day % 7

    def _evolve_continuous(
        self,
        state: StateView,
        action: str,
        next_activity: str,
    ) -> tuple[float | None, float | None, int | None, int | None]:
        """Evolve continuous state fields using state_dynamics config.

        Returns (new_steps, new_weight, new_time_of_day, new_day_of_week).
        If state_dynamics is None, returns (None, None, tod, dow).
        """
        tod = self._compute_time_of_day(state.step_of_day, state.steps_per_day)
        dow = self._compute_day_of_week(state.day)

        if self._state_dynamics is None:
            return None, None, tod, dow

        # Evolve steps
        sd = self._state_dynamics.steps
        resp_mult = sd.response_multiplier.get(action, 0.0)
        tod_mod = sd.tod_modulation.get(tod, 0.0)
        dow_mod = sd.dow_modulation.get(dow, 0.0)
        steps_mean = resp_mult * tod_mod * dow_mod
        steps_noise = self._rng.normal(0, sd.noise_std)
        new_steps = (state.steps or 0.0) + steps_mean + steps_noise

        # Evolve weight
        wd = self._state_dynamics.weight
        meal_effect = wd.meal_effect.get(tod, 0.0)
        weekend_boost = wd.weekend_boost if dow in {0, 6} else 0.0
        steps_delta = steps_mean + steps_noise
        weight_noise = self._rng.normal(0, wd.noise_std)
        new_weight = (
            (state.weight or 0.0)
            + meal_effect
            + weekend_boost
            + wd.steps_coefficient * steps_delta
            + weight_noise
        )

        return new_steps, new_weight, tod, dow

    def transition(self, state: StateView, action: str) -> StateView:
        # MVP mode: no state_dynamics
        if self._state_dynamics is None:
            next_activity = self._sample_activity(state, action)
            return dataclasses.replace(state, activity=next_activity)

        # Extended mode
        next_activity = self._sample_activity(state, action)
        new_steps, new_weight, tod, dow = self._evolve_continuous(
            state, action, next_activity
        )
        return dataclasses.replace(
            state,
            activity=next_activity,
            steps=new_steps,
            weight=new_weight,
            time_of_day=tod,
            day_of_week=dow,
        )


def register() -> None:
    from rl_health_interventions.transitions import REGISTRY

    REGISTRY["rule_based"] = RuleBasedTransition
