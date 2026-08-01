"""Unit tests for generate_pearl_mini temperature plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pearl_recalibration.generate_pearl_mini import (  # noqa: E402
    _resolve_temperature,
)


def test_resolve_temperature_defaults_to_batch_complete_default() -> None:
    assert _resolve_temperature(None) == 0.7


def test_resolve_temperature_passes_through_explicit_values() -> None:
    assert _resolve_temperature(0.3) == 0.3
    assert _resolve_temperature(0.0) == 0.0
    assert _resolve_temperature(1.2) == 1.2


def test_resolve_temperature_never_returns_none() -> None:
    for value in (None, 0.3, 0.7, 0.9):
        assert _resolve_temperature(value) is not None


def test_resolve_temperature_rejects_no_arg_form() -> None:
    with pytest.raises(TypeError):
        _resolve_temperature()  # type: ignore[call-arg]
