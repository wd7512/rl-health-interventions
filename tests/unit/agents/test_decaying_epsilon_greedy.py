from rl_health_interventions.agents.decaying_epsilon_greedy import (
    DecayingEpsilonGreedyAgent,
)


def test_select_action_returns_string_action(state_view):
    agent = DecayingEpsilonGreedyAgent(
        actions=["nudge", "idle"], epsilon_start=0.5, seed=42
    )
    action = agent.select_action(state_view)
    assert action in ("nudge", "idle")


def test_epsilon_decays_over_time(state_view):
    agent = DecayingEpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon_start=1.0,
        epsilon_min=0.0,
        decay_steps=100,
        seed=42,
    )
    for _ in range(50):
        agent.select_action(state_view)
    assert agent._current_epsilon() == 0.5
    for _ in range(50):
        agent.select_action(state_view)
    assert agent._current_epsilon() == 0.0


def test_epsilon_stays_above_min(state_view):
    agent = DecayingEpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon_start=0.5,
        epsilon_min=0.1,
        decay_steps=100,
        seed=42,
    )
    for _ in range(500):
        agent.select_action(state_view)
    assert agent._current_epsilon() == 0.1


def test_explore_rate_decreases_over_time(state_view):
    agent = DecayingEpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon_start=1.0,
        epsilon_min=0.0,
        decay_steps=200,
        seed=42,
    )
    agent.q_values["nudge"] = 10.0
    agent.q_values["idle"] = 0.0
    early = [agent.select_action(state_view) for _ in range(200)]
    late = [agent.select_action(state_view) for _ in range(200)]
    assert late.count("idle") / 200 < early.count("idle") / 200


def test_update_tracks_average_reward(state_view):
    agent = DecayingEpsilonGreedyAgent(
        actions=["nudge", "idle"], epsilon_start=0.5, seed=42
    )
    for _ in range(100):
        agent.update(state_view, "nudge", reward=1.0, next_state=state_view)
    assert agent.q_values["nudge"] == 1.0


def test_greedy_prefers_better_arm_after_learning(state_view):
    agent = DecayingEpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon_start=0.0,
        epsilon_min=0.0,
        decay_steps=100,
        seed=42,
    )
    for _ in range(50):
        agent.update(state_view, "nudge", reward=1.0, next_state=state_view)
        agent.update(state_view, "idle", reward=0.0, next_state=state_view)
    for _ in range(10):
        assert agent.select_action(state_view) == "nudge"


def test_seed_reproducibility(state_view):
    agent1 = DecayingEpsilonGreedyAgent(
        actions=["nudge", "idle"], epsilon_start=0.5, seed=99
    )
    agent2 = DecayingEpsilonGreedyAgent(
        actions=["nudge", "idle"], epsilon_start=0.5, seed=99
    )
    a1 = [agent1.select_action(state_view) for _ in range(100)]
    a2 = [agent2.select_action(state_view) for _ in range(100)]
    assert a1 == a2


def test_contextual_learns_per_context():
    from rl_health_interventions.state import StateView

    agent = DecayingEpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon_start=0.0,
        epsilon_min=0.0,
        decay_steps=100,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    sed = StateView({"activity": "sedentary"}, day=0, step_of_day=0)
    act = StateView({"activity": "active"}, day=0, step_of_day=0)
    for _ in range(100):
        agent.update(sed, "nudge", reward=0.0, next_state=sed)
        agent.update(sed, "idle", reward=1.0, next_state=sed)
        agent.update(act, "nudge", reward=1.0, next_state=act)
        agent.update(act, "idle", reward=0.0, next_state=act)
    assert (
        agent.q_values[("sedentary", "idle")] > agent.q_values[("sedentary", "nudge")]
    )
    assert agent.q_values[("active", "nudge")] > agent.q_values[("active", "idle")]


def test_epsilon_constant_when_min_equals_start(state_view):
    agent = DecayingEpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon_start=0.5,
        epsilon_min=0.5,
        decay_steps=200,
        seed=42,
    )
    for _ in range(1000):
        agent.select_action(state_view)
    assert agent._current_epsilon() == 0.5


def test_invalid_params_raise():
    import pytest

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        DecayingEpsilonGreedyAgent(epsilon_start=1.5)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        DecayingEpsilonGreedyAgent(epsilon_start=0.5, epsilon_min=1.5)
    with pytest.raises(ValueError, match="epsilon_min must not exceed"):
        DecayingEpsilonGreedyAgent(epsilon_start=0.1, epsilon_min=0.5)
    with pytest.raises(ValueError, match="decay_steps must be positive"):
        DecayingEpsilonGreedyAgent(epsilon_start=0.5, decay_steps=0)


def test_registration_in_registry():
    from rl_health_interventions.agents import REGISTRY

    assert "decaying_epsilon_greedy" in REGISTRY
    assert REGISTRY["decaying_epsilon_greedy"] is DecayingEpsilonGreedyAgent
