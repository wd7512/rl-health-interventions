from rl_health_interventions.config.schemas import Action
from rl_health_interventions.agents.epsilon_greedy import EpsilonGreedyAgent


def test_select_action_returns_valid_action():
    agent = EpsilonGreedyAgent(epsilon=0.1, seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    action = agent.select_action(state)
    assert action in (Action.SEND, Action.DON_T_SEND)


def test_explores_with_epsilon_probability():
    """With one clearly better arm, exploration picks suboptimal arm ~epsilon * 0.5."""
    agent = EpsilonGreedyAgent(epsilon=0.3, seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    agent.q_values[Action.SEND] = 10.0
    agent.q_values[Action.DON_T_SEND] = 0.0
    counts = {Action.SEND: 0, Action.DON_T_SEND: 0}
    for _ in range(10_000):
        action = agent.select_action(state)
        counts[action] += 1
    explore_rate = counts[Action.DON_T_SEND] / 10_000
    assert abs(explore_rate - 0.15) < 0.03


def test_update_tracks_average_reward():
    agent = EpsilonGreedyAgent(epsilon=0.1, seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    for _ in range(100):
        agent.update(state, Action.SEND, reward=1.0, next_state=state)
    assert agent.q_values[Action.SEND] == 1.0


def test_greedy_prefers_better_arm_after_learning():
    agent = EpsilonGreedyAgent(epsilon=0.0, seed=42)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    for _ in range(50):
        agent.update(state, Action.SEND, reward=1.0, next_state=state)
        agent.update(state, Action.DON_T_SEND, reward=0.0, next_state=state)
    for _ in range(10):
        assert agent.select_action(state) == Action.SEND


def test_seed_reproducibility():
    agent1 = EpsilonGreedyAgent(epsilon=0.3, seed=99)
    agent2 = EpsilonGreedyAgent(epsilon=0.3, seed=99)
    state = {"activity": "sedentary", "time_of_day": "morning"}
    actions1 = [agent1.select_action(state) for _ in range(100)]
    actions2 = [agent2.select_action(state) for _ in range(100)]
    assert actions1 == actions2
