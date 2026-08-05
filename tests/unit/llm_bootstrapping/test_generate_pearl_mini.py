"""Unit tests for generate_pearl_mini temperature plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl_health_interventions.llm_bootstrapping.request import (  # noqa: E402
    DEFAULT_TEMPERATURE,
)
from scripts.pearl_recalibration.generate_pearl_mini import (  # noqa: E402
    _resolve_temperature,
)


def test_resolve_temperature_defaults_to_batch_complete_default() -> None:
    assert _resolve_temperature(None) == DEFAULT_TEMPERATURE


def test_resolve_temperature_passes_through_explicit_values() -> None:
    assert _resolve_temperature(0.3) == 0.3
    assert _resolve_temperature(0.0) == 0.0
    assert _resolve_temperature(1.2) == 1.2


def test_request_forwards_resolved_temperature(monkeypatch) -> None:
    """The resolved temperature reaches batch_complete via the temperature kwarg."""
    from scripts.pearl_recalibration import generate_pearl_mini as pm

    captured: dict[str, float | None] = {}

    def fake_batch_complete(prompts, **kwargs):
        captured["temperature"] = kwargs["temperature"]
        captured["timeout"] = kwargs.get("timeout")
        return [{"content": "ok"}]

    monkeypatch.setattr(pm, "batch_complete", fake_batch_complete)
    result = pm._request(["p"], "sys", temperature=None, timeout=15.0)
    assert result == [{"content": "ok"}]
    assert captured["temperature"] == DEFAULT_TEMPERATURE
    assert captured["timeout"] == 15.0


def test_request_passes_explicit_temperature(monkeypatch) -> None:
    from scripts.pearl_recalibration import generate_pearl_mini as pm

    captured: dict[str, float | None] = {}

    def fake_batch_complete(prompts, **kwargs):
        captured["temperature"] = kwargs["temperature"]
        return [{"content": "ok"}]

    monkeypatch.setattr(pm, "batch_complete", fake_batch_complete)
    pm._request(["p"], "sys", temperature=0.3, timeout=None)
    assert captured["temperature"] == 0.3
