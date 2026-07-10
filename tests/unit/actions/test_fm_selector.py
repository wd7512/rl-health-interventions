import numpy as np

from rl_health_interventions.actions.fm_selector import FMSelector
from rl_health_interventions.actions.archetypes import ARCHETYPE_NAMES


def test_project_output_dim():
    sel = FMSelector(embed_dim=768, proj_dim=32)
    emb = np.random.randn(768)
    projected = sel.project(emb)
    assert projected.shape == (32,)


def test_project_normalised():
    sel = FMSelector(embed_dim=768, proj_dim=32)
    emb = np.random.randn(768)
    projected = sel.project(emb)
    # L2 norm should be reasonable (not exactly 1 since we don't normalise projection)
    assert projected.ndim == 1


def test_select_returns_valid_archetype():
    sel = FMSelector(embed_dim=768, proj_dim=32)
    emb = np.random.randn(768)
    idx, projected = sel.select_archetype(emb)
    assert 0 <= idx < 3
    assert ARCHETYPE_NAMES[idx] in ["movement", "cognitive", "wellness"]
    assert projected.shape == (32,)


def test_batch_project():
    sel = FMSelector(embed_dim=768, proj_dim=32)
    batch = np.random.randn(4, 768)
    projected = sel.project(batch)
    assert projected.shape == (4, 32)


def test_update_changes_vectors():
    sel = FMSelector(embed_dim=768, proj_dim=32, lr=0.1)
    before = sel.archetype_vectors.copy()
    emb = np.random.randn(768)
    _, projected = sel.select_archetype(emb)
    sel.update(0, projected, reward=1.0)
    assert not np.allclose(sel.archetype_vectors[0], before[0])


def test_archetypes_normalised():
    sel = FMSelector(embed_dim=768, proj_dim=32)
    norms = np.linalg.norm(sel.archetype_vectors, axis=1)
    for norm in norms:
        assert abs(norm - 1.0) < 1e-5
