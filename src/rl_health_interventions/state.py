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
    _state_key: tuple[str, ...]
    _factor_names: tuple[str, ...]
    _hash_value: int

    def __init__(
        self,
        factors: dict[str, str],
        day: int = 0,
        step_of_day: int = 0,
        steps_per_day: int = 5,
    ):
        sorted_items = sorted(factors.items(), key=lambda x: x[0])
        object.__setattr__(self, "_state_key", tuple(v for _, v in sorted_items))
        object.__setattr__(self, "_factor_names", tuple(factors.keys()))
        object.__setattr__(self, "_factors", factors)
        object.__setattr__(self, "_day", day)
        object.__setattr__(self, "_step_of_day", step_of_day)
        object.__setattr__(self, "_steps_per_day", steps_per_day)
        object.__setattr__(
            self,
            "_hash_value",
            hash((self._state_key, day, step_of_day, steps_per_day)),
        )

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
        return self._state_key

    @property
    def factor_names(self) -> tuple[str, ...]:
        return self._factor_names

    @property
    def factor_values(self) -> dict[str, str]:
        return self._factors

    def with_factors(self, **updates) -> StateView:
        unknown = set(updates.keys()) - set(self._factor_names)
        if unknown:
            raise ValueError(
                f"Unknown factor updates: {sorted(unknown)}. "
                f"Available factors: {sorted(self._factor_names)}"
            )
        new = dict(self._factors)
        new.update(updates)
        return StateView(new, self._day, self._step_of_day, self._steps_per_day)

    @staticmethod
    def _from_parts(
        factors: dict[str, str],
        day: int,
        step_of_day: int,
        steps_per_day: int,
        state_key: tuple[str, ...],
        factor_names: tuple[str, ...],
    ) -> StateView:
        sv = object.__new__(StateView)
        object.__setattr__(sv, "_factors", factors)
        object.__setattr__(sv, "_day", day)
        object.__setattr__(sv, "_step_of_day", step_of_day)
        object.__setattr__(sv, "_steps_per_day", steps_per_day)
        object.__setattr__(sv, "_state_key", state_key)
        object.__setattr__(sv, "_factor_names", factor_names)
        object.__setattr__(
            sv,
            "_hash_value",
            hash((state_key, day, step_of_day, steps_per_day)),
        )
        return sv

    def with_advance(
        self, day: int | None = None, step_of_day: int | None = None
    ) -> StateView:
        """Return new StateView with updated day/step."""
        new_day = self._day if day is None else day
        new_step = self._step_of_day if step_of_day is None else step_of_day
        if new_day == self._day and new_step == self._step_of_day:
            return self
        return StateView._from_parts(
            self._factors,
            new_day,
            new_step,
            self._steps_per_day,
            self._state_key,
            self._factor_names,
        )

    def __eq__(self, other):
        if not isinstance(other, StateView):
            return False
        return (
            self._state_key == other._state_key
            and self._day == other._day
            and self._step_of_day == other._step_of_day
            and self._steps_per_day == other._steps_per_day
        )

    def __hash__(self):
        return self._hash_value

    def __repr__(self):
        parts = ", ".join(f"{k}={v}" for k, v in self._factors.items())
        return f"StateView({parts}, day={self._day}, step_of_day={self._step_of_day})"
