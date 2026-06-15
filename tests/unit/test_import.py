from __future__ import annotations

import pytest

from rl_health_interventions.__main__ import main


def test_main_callable() -> None:
    assert callable(main)


def test_main_exits_without_config() -> None:
    """Calling main() without --config should exit with code 2."""
    with pytest.raises(SystemExit, match="2"):
        main()
