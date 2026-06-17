from rl_health_interventions.agents.epsilon_greedy import EpsilonGreedyAgent
from rl_health_interventions.state import StateView


def _sv() -> StateView:
    return StateView(activity="sedentary", day=0, step_of_day=0)


def test_select_action_returns_string_action():
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.1, seed=42)
    action = agent.select_action(_sv())
    assert action in ("nudge", "idle")


def test_explores_with_epsilon_probability():
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.3, seed=42)
    state = _sv()
    agent.q_values["nudge"] = 10.0
    agent.q_values["idle"] = 0.0
    counts = {"nudge": 0, "idle": 0}
    for _ in range(10_000):
        action = agent.select_action(state)
        counts[action] += 1
    explore_rate = counts["idle"] / 10_000
    assert abs(explore_rate - 0.15) < 0.03


def test_update_tracks_average_reward():
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.1, seed=42)
    state = _sv()
    for _ in range(100):
        agent.update(state, "nudge", reward=1.0, next_state=state)
    assert agent.q_values["nudge"] == 1.0


def test_greedy_prefers_better_arm_after_learning():
    agent = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.0, seed=42)
    state = _sv()
    for _ in range(50):
        agent.update(state, "nudge", reward=1.0, next_state=state)
        agent.update(state, "idle", reward=0.0, next_state=state)
    for _ in range(10):
        assert agent.select_action(state) == "nudge"


def test_seed_reproducibility():
    agent1 = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.3, seed=99)
    agent2 = EpsilonGreedyAgent(actions=["nudge", "idle"], epsilon=0.3, seed=99)
    state = _sv()
    actions1 = [agent1.select_action(state) for _ in range(100)]
    actions2 = [agent2.select_action(state) for _ in range(100)]
    assert actions1 == actions2
