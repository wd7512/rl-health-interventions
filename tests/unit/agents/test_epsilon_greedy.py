from rl_health_interventions.agents.contextual_bandits.epsilon_greedy import (
    EpsilonGreedyAgent,
)


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


def test_contextual_epsilon_greedy_learns_per_context(sed_and_act):
    sed, act = sed_and_act
    agent = EpsilonGreedyAgent(
        actions=["nudge", "idle"],
        epsilon=0.0,
        seed=42,
        contextual=True,
        context_features="activity_level",
    )

    for _ in range(100):
        agent.update(sed, "nudge", reward=0.0, next_state=sed)
        agent.update(sed, "idle", reward=1.0, next_state=sed)
        agent.update(act, "nudge", reward=1.0, next_state=act)
        agent.update(act, "idle", reward=0.0, next_state=act)

    assert (
        agent.q_values[("sedentary", "idle")] > agent.q_values[("sedentary", "nudge")]
    )
    assert agent.q_values[("active", "nudge")] > agent.q_values[("active", "idle")]
