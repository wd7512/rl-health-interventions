from rl_health_interventions.actions.archetypes import (
    ALL_INTERVENTIONS,
    ARCHETYPES,
    ARCHETYPE_NAMES,
    ACTION_TO_ARCHETYPE,
    get_intervention_by_name,
    get_intervention_names,
)


def test_total_intervention_count():
    assert len(ALL_INTERVENTIONS) == 15


def test_archetype_count():
    assert len(ARCHETYPES) == 3


def test_each_archetype_has_5():
    for name in ARCHETYPE_NAMES:
        assert len(ARCHETYPES[name]) == 5


def test_all_names_unique():
    names = [i.name for i in ALL_INTERVENTIONS]
    assert len(names) == len(set(names))


def test_lookup_by_name():
    for i in ALL_INTERVENTIONS:
        assert get_intervention_by_name(i.name) is i


def test_get_intervention_names():
    names = get_intervention_names()
    assert len(names) == 15
    assert all(isinstance(n, str) for n in names)


def test_action_to_archetype_mapping():
    for i in ALL_INTERVENTIONS:
        assert ACTION_TO_ARCHETYPE[i.name] == i.archetype


def test_cost_range():
    for i in ALL_INTERVENTIONS:
        assert 0.0 <= i.base_cost <= 0.1
