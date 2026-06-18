from rl_health_interventions.agents.epsilon_greedy import EpsilonGreedyAgent


def test_select_action_returns_string_action(state_view):
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.1, seed=42)
    action = agent.select_action(state_view)
    assert action in ("nudge", "idle")


def test_explores_with_epsilon_probability(state_view):
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.3, seed=42)
    agent.q_values["nudge"] = 10.0
    agent.q_values["idle"] = 0.0
    counts = {"nudge": 0, "idle": 0}
    for _ in range(10_000):
        action = agent.select_action(state_view)
        counts[action] += 1
    explore_rate = counts["idle"] / 10_000
    assert abs(explore_rate - 0.15) < 0.03


def test_update_tracks_average_reward(state_view):
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.1, seed=42)
    for _ in range(100):
        agent.update(state_view, "nudge", reward=1.0, next_state=state_view)
    assert agent.q_values["nudge"] == 1.0


def test_greedy_prefers_better_arm_after_learning(state_view):
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.0, seed=42)
    for _ in range(50):
        agent.update(state_view, "nudge", reward=1.0, next_state=state_view)
        agent.update(state_view, "idle", reward=0.0, next_state=state_view)
    for _ in range(10):
        assert agent.select_action(state_view) == "nudge"


def test_seed_reproducibility(state_view):
    agent1 = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.3, seed=99)
    agent2 = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.3, seed=99)
    actions1 = [agent1.select_action(state_view) for _ in range(100)]
    actions2 = [agent2.select_action(state_view) for _ in range(100)]
    assert actions1 == actions2


def test_contextual_epsilon_greedy_learns_per_context():
    from rl_health_interventions.state import StateView

    agent = EpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon=0.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )

    sed = StateView(activity="sedentary", day=0, step_of_day=0)
    act = StateView(activity="active", day=0, step_of_day=0)

    for _ in range(100):
        agent.update(sed, "nudge", reward=0.0, next_state=sed)
        agent.update(sed, "idle", reward=1.0, next_state=sed)
        agent.update(act, "nudge", reward=1.0, next_state=act)
        agent.update(act, "idle", reward=0.0, next_state=act)

    assert (
        agent.q_values[("sedentary", "idle")] > agent.q_values[("sedentary", "nudge")]
    )
    assert agent.q_values[("active", "nudge")] > agent.q_values[("active", "idle")]


# ---------------------------------------------------------------------------
# Epsilon validation – boundary and invalid values
# ---------------------------------------------------------------------------


def test_epsilon_zero_is_valid():
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.0, seed=42)
    assert agent.epsilon == 0.0


def test_epsilon_one_is_valid():
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=1.0, seed=42)
    assert agent.epsilon == 1.0


def test_epsilon_negative_raises():
    import pytest

    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedyAgent(epsilon=-0.01)


def test_epsilon_above_one_raises():
    import pytest

    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedyAgent(epsilon=1.01)


# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------


def test_non_contextual_init_prepopulates_q_values():
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.0, seed=42)
    assert "nudge" in agent.q_values
    assert "idle" in agent.q_values
    assert agent.q_values["nudge"] == 0.0
    assert agent.q_values["idle"] == 0.0


def test_non_contextual_init_prepopulates_counts():
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.0, seed=42)
    assert agent.counts["nudge"] == 0
    assert agent.counts["idle"] == 0


def test_contextual_init_q_values_start_empty():
    agent = EpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon=0.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    assert len(agent.q_values) == 0


def test_contextual_init_counts_start_empty():
    agent = EpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon=0.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    assert len(agent.counts) == 0


# ---------------------------------------------------------------------------
# _ensure_params – lazy initialisation
# ---------------------------------------------------------------------------


def test_ensure_params_creates_key_on_first_call():
    agent = EpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon=0.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    agent._ensure_params(("sedentary", "nudge"))
    assert ("sedentary", "nudge") in agent.q_values
    assert agent.q_values[("sedentary", "nudge")] == 0.0
    assert agent.counts[("sedentary", "nudge")] == 0


def test_ensure_params_is_idempotent():
    import pytest

    agent = EpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon=0.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    key = ("sedentary", "nudge")
    agent._ensure_params(key)
    agent.q_values[key] = 5.0
    agent._ensure_params(key)  # second call must not reset
    assert agent.q_values[key] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# select_action behaviour
# ---------------------------------------------------------------------------


def test_epsilon_one_always_returns_valid_action(state_view):
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=1.0, seed=42)
    for _ in range(50):
        action = agent.select_action(state_view)
        assert action in ("nudge", "idle")


def test_contextual_select_before_updates_returns_valid_action():
    """Before any updates q_values are lazily created; select_action must still work."""
    from rl_health_interventions.state import StateView

    agent = EpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon=0.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    state = StateView(activity="sedentary", day=0, step_of_day=0)
    action = agent.select_action(state)
    assert action in ("nudge", "idle")


# ---------------------------------------------------------------------------
# Update isolation between contexts
# ---------------------------------------------------------------------------


def test_updating_one_context_does_not_affect_another():
    import pytest
    from rl_health_interventions.state import StateView

    agent = EpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon=0.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    sed = StateView(activity="sedentary", day=0, step_of_day=0)
    act = StateView(activity="active", day=0, step_of_day=0)

    for _ in range(50):
        agent.update(sed, "nudge", reward=1.0, next_state=sed)

    # active context keys should not yet exist
    assert ("active", "nudge") not in agent.q_values
    assert ("active", "idle") not in agent.q_values

    # sedentary nudge should have been updated
    assert agent.q_values[("sedentary", "nudge")] == pytest.approx(1.0)

    # now update active and confirm isolation
    for _ in range(50):
        agent.update(act, "idle", reward=0.5, next_state=act)

    assert agent.q_values[("active", "idle")] == pytest.approx(0.5)
    assert agent.q_values[("sedentary", "nudge")] == pytest.approx(1.0)  # unchanged
