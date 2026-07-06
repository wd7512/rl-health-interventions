from __future__ import annotations

from rl_health_interventions.config.schemas import MDPConfig
from rl_health_interventions.environment import Environment, _bin_daily


def _sprint1_env_config(seed: int = 42) -> MDPConfig:
    return MDPConfig(
        episode_days=2,
        steps_per_day=3,
        seed=seed,
        state={
            "variables": {
                "step_bin": {"names": ["inactive", "moderate", "active"]},
                "sleep": {"names": ["good", "poor"]},
                "day_of_week": {
                    "names": ["weekday", "weekend"],
                    "advanced": {
                        "type": "cyclic",
                        "granularity": "daily",
                        "pattern": ["weekday", "weekend"],
                    },
                },
                "burden": {
                    "names": ["low", "medium", "high"],
                    "advanced": {
                        "type": "rolling_window_count",
                        "window_size": 3,
                        "conditions": [
                            {"factor": "action", "type": "in", "values": ["go"]}
                        ],
                        "mapping": {0: "low", 1: "medium", 2: "high", 3: "high"},
                    },
                },
            }
        },
        initial_state={
            "step_bin": "inactive",
            "sleep": "good",
            "day_of_week": "weekday",
            "burden": "low",
        },
        actions=["idle", "go"],
        reward={
            "variables": {
                "v": {
                    "source": "state.step_bin",
                    "mapping": {"inactive": 0.0, "moderate": 0.5, "active": 1.0},
                }
            },
            "formula": "v",
        },
        transition_model={"type": "random"},
        agents=[],
    )


class TestBinDaily:
    def test_inactive(self) -> None:
        assert _bin_daily(0) == "inactive"
        assert _bin_daily(3999) == "inactive"

    def test_moderate(self) -> None:
        assert _bin_daily(4000) == "moderate"
        assert _bin_daily(8000) == "moderate"

    def test_active(self) -> None:
        assert _bin_daily(8001) == "active"
        assert _bin_daily(20000) == "active"


class TestDayBoundary:
    def test_step_zero_injects_step_bin_daily(self) -> None:
        env = Environment(_sprint1_env_config(seed=42), seed=42)
        env.reset()
        # Run 4 steps: day 0 (steps 0-2), then step 3 starts day 1 (step_idx=0)
        for _ in range(4):
            state, _, _ = env.step("go")
        assert hasattr(state, "step_bin_daily")
        assert state.step_bin_daily in ("inactive", "moderate", "active")

    def test_daily_total_resets_at_boundary(self) -> None:
        env = Environment(_sprint1_env_config(seed=42), seed=42)
        env.reset()
        # Run 3 steps to accumulate daily_total (day 0)
        for _ in range(3):
            env.step("go")
        prev_total = env._daily_total
        assert prev_total > 0
        # Step 4 triggers day boundary (step_idx=0), resets daily_total
        # After reset, _update_daily_total adds the new step_bin's midpoint
        env.step("go")
        assert env._daily_total < prev_total


class TestCyclicAdvance:
    def test_day_of_week_follows_pattern(self) -> None:
        config = _sprint1_env_config(seed=42)
        env = Environment(config, seed=42)
        state = env.reset()
        assert state.day_of_week == "weekday"  # day 0, pattern[0]
        # Run 4 steps: day 0 ends at step 3, day 1 starts
        # After step 4: day=1, step_of_day=1, cyclic advance applied with day=1
        for _ in range(4):
            state, _, _ = env.step("go")
        assert state.day == 1
        assert state.day_of_week == "weekend"  # day 1, pattern[1]


class TestRollingAdvance:
    def test_burden_increases_with_go_actions(self) -> None:
        config = _sprint1_env_config(seed=42)
        env = Environment(config, seed=42)
        state = env.reset()
        assert state.burden == "low"  # primed with 3 idles
        # 3 "go" actions fill the window
        for _ in range(3):
            state, _, _ = env.step("go")
        assert state.burden == "high"
