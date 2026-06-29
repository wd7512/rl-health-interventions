from __future__ import annotations


class StateView:
    """Immutable factored state for the MDP.

    Factor values are stored in a dict and accessible via __getattr__.
    This supports both sprint1's 4-factor state and MVP's single-factor state.
    """

    _factors: dict[str, str]
    _day: int
    _step_of_day: int
    _steps_per_day: int

    def __init__(
        self,
        factors: dict[str, str],
        day: int = 0,
        step_of_day: int = 0,
        steps_per_day: int = 5,
    ):
        object.__setattr__(self, "_factors", dict(factors))
        object.__setattr__(self, "_day", day)
        object.__setattr__(self, "_step_of_day", step_of_day)
        object.__setattr__(self, "_steps_per_day", steps_per_day)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError(f"'StateView' is immutable: cannot set attribute '{name}'")

    @property
    def day(self) -> int:
        return self._day

    @property
    def step_of_day(self) -> int:
        return self._step_of_day

    @property
    def steps_per_day(self) -> int:
        return self._steps_per_day

    def __getattr__(self, name: str) -> str:
        try:
            return self._factors[name]
        except KeyError:
            raise AttributeError(
                f"'StateView' has no attribute '{name}'. "
                f"Available factors: {list(self._factors.keys())}"
            )

    @property
    def global_step(self) -> int:
        return self._day * self._steps_per_day + self._step_of_day

    @property
    def state_key(self) -> tuple[str, ...]:
        return tuple(v for _, v in sorted(self._factors.items(), key=lambda x: x[0]))

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(self._factors.keys())

    def with_factors(self, **updates) -> StateView:
        unknown = set(updates.keys()) - set(self._factors.keys())
        if unknown:
            raise ValueError(
                f"Unknown factor updates: {sorted(unknown)}. "
                f"Available factors: {sorted(self._factors.keys())}"
            )
        new = dict(self._factors)
        new.update(updates)
        return StateView(new, self._day, self._step_of_day, self._steps_per_day)

    def with_advance(
        self, day: int | None = None, step_of_day: int | None = None
    ) -> StateView:
        """Return new StateView with updated day/step."""
        return StateView(
            dict(self._factors),
            day if day is not None else self._day,
            step_of_day if step_of_day is not None else self._step_of_day,
            self._steps_per_day,
        )

    def __eq__(self, other):
        if not isinstance(other, StateView):
            return False
        return (
            self._factors == other._factors
            and self._day == other._day
            and self._step_of_day == other._step_of_day
            and self._steps_per_day == other._steps_per_day
        )

    def __hash__(self):
        return hash(
            (
                frozenset(self._factors.items()),
                self._day,
                self._step_of_day,
                self._steps_per_day,
            )
        )

    def __repr__(self):
        parts = ", ".join(f"{k}={v}" for k, v in self._factors.items())
        return f"StateView({parts}, day={self._day}, step_of_day={self._step_of_day})"
