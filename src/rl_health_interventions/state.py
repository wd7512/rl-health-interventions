from __future__ import annotations

from typing_extensions import override


class StateView:
    __slots__ = (
        "_factors",
        "_day",
        "_step_of_day",
        "_steps_per_day",
        "_hash",
    )

    _factors: dict[str, str]
    _day: int
    _step_of_day: int
    _steps_per_day: int
    _hash: int | None

    def __init__(
        self,
        factors: dict[str, str],
        day: int = 0,
        step_of_day: int = 0,
        steps_per_day: int = 5,
    ) -> None:
        object.__setattr__(self, "_factors", dict(factors))
        object.__setattr__(self, "_day", day)
        object.__setattr__(self, "_step_of_day", step_of_day)
        object.__setattr__(self, "_steps_per_day", steps_per_day)
        object.__setattr__(self, "_hash", None)

    def __getattr__(self, name: str) -> str:
        try:
            return self._factors[name]
        except KeyError:
            msg = (
                f"StateView has no factor '{name}'. "
                f"Available factors: {sorted(self._factors)}"
            )
            raise AttributeError(msg) from None

    @override
    def __setattr__(self, name: str, value: object) -> None:
        msg = "StateView is immutable"
        raise AttributeError(msg)

    @property
    def day(self) -> int:
        return self._day

    @property
    def step_of_day(self) -> int:
        return self._step_of_day

    @property
    def steps_per_day(self) -> int:
        return self._steps_per_day

    @property
    def global_step(self) -> int:
        return self._day * self._steps_per_day + self._step_of_day

    @property
    def factor_values(self) -> dict[str, str]:
        return dict(self._factors)

    @property
    def state_key(self) -> tuple[str, ...]:
        return tuple(self._factors.get(k, "") for k in sorted(self._factors))

    def with_factors(self, **updates: str) -> StateView:
        new_factors = {**self._factors, **updates}
        return StateView(new_factors, self._day, self._step_of_day, self._steps_per_day)

    def with_advance(self) -> StateView:
        """Advance to the next timestep. Called by the environment after a step."""
        step = self._step_of_day + 1
        next_step_of_day = step % self._steps_per_day
        next_day = self._day + (1 if next_step_of_day == 0 else 0)
        return StateView(self._factors, next_day, next_step_of_day, self._steps_per_day)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateView):
            return NotImplemented
        return (
            self._factors == other._factors
            and self._day == other._day
            and self._step_of_day == other._step_of_day
            and self._steps_per_day == other._steps_per_day
        )

    @override
    def __hash__(self) -> int:
        cached_hash = self._hash
        if cached_hash is None:
            cached_hash = hash(
                (
                    tuple(sorted(self._factors.items())),
                    self._day,
                    self._step_of_day,
                    self._steps_per_day,
                )
            )
            object.__setattr__(self, "_hash", cached_hash)
        return cached_hash

    @override
    def __repr__(self) -> str:
        factors_repr = ", ".join(f"{k}={v!r}" for k, v in sorted(self._factors.items()))
        return (
            f"StateView({factors_repr},"
            f" day={self._day}, step_of_day={self._step_of_day})"
        )
