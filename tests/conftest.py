from __future__ import annotations

import pathlib

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-apply markers based on test file location.

    Respects explicit markers — if a test already has a marker, don't override.
    """
    unit_dir = (pathlib.Path(__file__).parent / "unit").resolve()
    integration_dir = (pathlib.Path(__file__).parent / "integration").resolve()
    for item in items:
        test_path = pathlib.Path(item.path).resolve()
        existing_markers = {m.name for m in item.iter_markers()}
        if test_path.is_relative_to(unit_dir) and "integration" not in existing_markers:
            item.add_marker(pytest.mark.unit)
        elif (
            test_path.is_relative_to(integration_dir) and "unit" not in existing_markers
        ):
            item.add_marker(pytest.mark.integration)
