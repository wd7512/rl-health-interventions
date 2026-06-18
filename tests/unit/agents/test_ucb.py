from rl_health_interventions.agents.ucb import UCBAgent


def test_select_action_returns_string_action(state_view):
    agent = UCBAgent(actions=["nudge", "idle"], seed=42)
    action = agent.select_action(state_view)
    assert action in ("nudge", "idle")


def test_initial_exploration_pulls_each_action(state_view):
    agent = UCBAgent(actions=["nudge", "idle"], seed=42)
    first = agent.select_action(state_view)
    agent.update(state_view, first, 1.0, state_view)
    second = agent.select_action(state_view)
    agent.update(state_view, second, 0.0, state_view)
    assert first != second
    assert agent.counts["nudge"] == 1
    assert agent.counts["idle"] == 1


def test_prefers_better_arm_after_learning(state_view):
    agent = UCBAgent(actions=["nudge", "idle"], c=1.0, seed=42)
    agent.update(state_view, "nudge", 1.0, state_view)
    agent.update(state_view, "idle", 0.0, state_view)
    for _ in range(100):
        agent.update(state_view, "nudge", 1.0, state_view)
    for _ in range(100):
        agent.update(state_view, "idle", 0.0, state_view)
    for _ in range(20):
        assert agent.select_action(state_view) == "nudge"


def test_ucb_exploration_bonus_decays(state_view):
    agent = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=42)
    agent.update(state_view, "nudge", 1.0, state_view)
    agent.update(state_view, "idle", 0.0, state_view)
    for _ in range(200):
        agent.update(state_view, "nudge", 1.0, state_view)
    for _ in range(200):
        agent.update(state_view, "idle", 0.0, state_view)
    for _ in range(20):
        assert agent.select_action(state_view) == "nudge"


def test_seed_reproducibility(state_view):
    agent1 = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=99)
    agent2 = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=99)
    for a in agent1, agent2:
        a.update(state_view, "nudge", 1.0, state_view)
        a.update(state_view, "idle", 0.0, state_view)
    actions1 = [agent1.select_action(state_view) for _ in range(100)]
    actions2 = [agent2.select_action(state_view) for _ in range(100)]
    assert actions1 == actions2


def test_c_validation():
    import pytest

    with pytest.raises(ValueError, match="strictly positive"):
        UCBAgent(c=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        UCBAgent(c=-1.0)


def test_contextual_ucb_learns_per_context():
    from rl_health_interventions.state import StateView

    agent = UCBAgent(
        actions=["nudge", "idle"],
        c=0.1,
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
# Parameter initialisation
# ---------------------------------------------------------------------------


def test_non_contextual_init_prepopulates_q_values():
    agent = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=42)
    assert "nudge" in agent.q_values
    assert "idle" in agent.q_values
    assert agent.q_values["nudge"] == 0.0
    assert agent.q_values["idle"] == 0.0


def test_non_contextual_init_prepopulates_counts():
    agent = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=42)
    assert agent.counts["nudge"] == 0
    assert agent.counts["idle"] == 0


def test_contextual_init_q_values_start_empty():
    agent = UCBAgent(
        actions=["nudge", "idle"],
        c=2.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    assert len(agent.q_values) == 0


def test_contextual_init_counts_start_empty():
    agent = UCBAgent(
        actions=["nudge", "idle"],
        c=2.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    assert len(agent.counts) == 0


# ---------------------------------------------------------------------------
# _total_steps counter
# ---------------------------------------------------------------------------


def test_total_steps_starts_at_zero():
    agent = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=42)
    assert agent._total_steps == 0


def test_total_steps_increments_on_update(state_view):
    agent = UCBAgent(actions=["nudge", "idle"], c=2.0, seed=42)
    agent.update(state_view, "nudge", reward=1.0, next_state=state_view)
    assert agent._total_steps == 1
    agent.update(state_view, "idle", reward=0.0, next_state=state_view)
    assert agent._total_steps == 2


def test_total_steps_global_across_contexts():
    """In contextual mode _total_steps is still a single global counter."""
    from rl_health_interventions.state import StateView

    agent = UCBAgent(
        actions=["nudge", "idle"],
        c=2.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    sed = StateView(activity="sedentary", day=0, step_of_day=0)
    act = StateView(activity="active", day=0, step_of_day=0)

    agent.update(sed, "nudge", reward=1.0, next_state=sed)
    agent.update(act, "idle", reward=0.0, next_state=act)
    assert agent._total_steps == 2


# ---------------------------------------------------------------------------
# _ensure_params – lazy initialisation
# ---------------------------------------------------------------------------


def test_ensure_params_creates_key_on_first_call():
    agent = UCBAgent(
        actions=["nudge", "idle"],
        c=2.0,
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

    agent = UCBAgent(
        actions=["nudge", "idle"],
        c=2.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    key = ("sedentary", "nudge")
    agent._ensure_params(key)
    agent.q_values[key] = 7.0
    agent._ensure_params(key)  # must not reset
    assert agent.q_values[key] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Contextual initial exploration
# ---------------------------------------------------------------------------


def test_contextual_initial_exploration_visits_unvisited_actions():
    """In contextual mode, UCB must try each unvisited (context, action) pair."""
    from rl_health_interventions.state import StateView

    agent = UCBAgent(
        actions=["nudge", "idle"],
        c=2.0,
        seed=42,
        contextual=True,
        context_feature="activity",
    )
    state = StateView(activity="sedentary", day=0, step_of_day=0)

    first = agent.select_action(state)
    agent.update(state, first, reward=1.0, next_state=state)

    second = agent.select_action(state)
    agent.update(state, second, reward=0.0, next_state=state)

    # Both actions must have been tried before UCB exploration kicks in
    assert agent.counts[("sedentary", "nudge")] == 1
    assert agent.counts[("sedentary", "idle")] == 1


# ---------------------------------------------------------------------------
# Regression: backward-compat import from old module path
# ---------------------------------------------------------------------------


def test_import_from_agents_module():
    """UCBAgent is still importable from the old top-level path."""
    from rl_health_interventions.agents.ucb import UCBAgent as UCBA  # noqa: F401

    assert UCBA is UCBAgent
