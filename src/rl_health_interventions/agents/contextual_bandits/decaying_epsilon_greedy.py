from __future__ import annotations

from rl_health_interventions.agents.contextual_bandits._base import (
    ContextualBanditAgent,
)


class DecayingEpsilonGreedyAgent(ContextualBanditAgent):
    """Epsilon-greedy with linearly decaying exploration rate.

    Epsilon decays each step from ``epsilon_start`` toward 0 following:
        epsilon = max(epsilon_min, epsilon_start * (1 - step / decay_steps))

    When ``contextual=True``, maintains separate ``(q_value, count)``
    entries for each ``(context_value, action)`` pair.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        epsilon_start: float = 0.5,
        epsilon_min: float = 0.01,
        decay_steps: int = 200,
        seed: int = 42,
        contextual: bool = False,
        context_feature: str | None = None,
    ) -> None:
        """
        Initialize a DecayingEpsilonGreedyAgent.

        Parameters:
            actions: Available action names. Passed to the base class.
            epsilon_start: Initial exploration probability, in [0.0, 1.0].
            epsilon_min: Minimum exploration probability, in [0.0, 1.0].
            decay_steps: Number of steps over which epsilon decays linearly.
            seed: RNG seed for reproducibility.
            contextual: If True, maintain separate Q-values per (context, action).
            context_feature: Name of the context feature column (required when contextual=True).

        Raises:
            ValueError: If epsilon_start or epsilon_min are not in [0.0, 1.0],
                if epsilon_min exceeds epsilon_start, or if decay_steps is not positive.
        """
        super().__init__(
            actions=actions,
            seed=seed,
            contextual=contextual,
            context_feature=context_feature,
        )
        if not (0.0 <= epsilon_start <= 1.0):
            raise ValueError("epsilon_start must be between 0.0 and 1.0 inclusive.")
        if not (0.0 <= epsilon_min <= 1.0):
            raise ValueError("epsilon_min must be between 0.0 and 1.0 inclusive.")
        if epsilon_min > epsilon_start:
            raise ValueError("epsilon_min must not exceed epsilon_start.")
        if decay_steps <= 0:
            raise ValueError("decay_steps must be positive.")
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_steps = decay_steps
        self._step = 0
        self._init_params()

    def _init_params(self) -> None:
        """
        Initialize Q-value and count storage.

        In non-contextual mode, pre-populates all actions with initial
        Q-values of 0.0 and visit counts of 0. In contextual mode,
        initializes empty dictionaries for lazy population.
        """
        self.q_values: dict = {}
        self.counts: dict = {}
        if not self.contextual:
            self.q_values = {a: 0.0 for a in self._actions}
            self.counts = {a: 0 for a in self._actions}

    def _ensure_params(self, key: str | tuple[str, ...]) -> None:
        """
        Ensure Q-value and count entries are initialized for the given key.

        Parameters:
            key (str | tuple[str, str]): The parameter key; either an action string or a (context_value, action) tuple.
        """
        if key not in self.q_values:
            self.q_values[key] = 0.0
            self.counts[key] = 0

    def _current_epsilon(self) -> float:
        """
        Compute the current exploration rate with linear decay.

        Returns:
            float: The exploration rate (epsilon) for the current step, between epsilon_min and epsilon_start.
        """
        decayed = self.epsilon_start * (1 - self._step / self.decay_steps)
        return max(self.epsilon_min, decayed)

    def select_action(self, state) -> str:
        """
        Select an action according to the epsilon-greedy policy.

        With probability epsilon, explores by selecting a uniformly random action.
        Otherwise, exploits by selecting uniformly at random among actions with the
        maximum Q-value.

        Parameters:
            state: Current environment state (used for contextual Q-value lookup).

        Returns:
            The selected action name.
        """
        epsilon = self._current_epsilon()
        self._step += 1
        if self._rng.random() < epsilon:
            idx = self._rng.integers(len(self._actions))
            return self._actions[idx]
        action_values = {}
        for action in self._actions:
            key = self._get_context_key(state, action)
            self._ensure_params(key)
            action_values[action] = self.q_values[key]
        max_q = max(action_values.values())
        best = [a for a, q in action_values.items() if q == max_q]
        idx = self._rng.integers(len(best))
        return best[idx]

    def update(self, state, action: str, reward: float, next_state) -> None:
        """
        Update the Q-value estimate for a state-action pair.

        Applies the incremental mean update: Q(s,a) += (reward - Q(s,a)) / n.

        Parameters:
            state: Current environment state.
            action: The action that was taken.
            reward: Scalar reward received from the environment.
            next_state: Successor state (unused by this agent).
        """
        key = self._get_context_key(state, action)
        self._ensure_params(key)
        self.counts[key] += 1
        n = self.counts[key]
        self.q_values[key] += (reward - self.q_values[key]) / n
