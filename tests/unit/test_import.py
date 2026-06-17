from __future__ import annotations

import sys

from rl_health_interventions.__main__ import main


def test_main_callable() -> None:
    assert callable(main)


def test_main_runs_with_default_config() -> None:
    """Calling main() without --config uses default and succeeds."""
    old_argv = sys.argv
    try:
        sys.argv = ["rl-health-interventions"]
        main()
    finally:
        sys.argv = old_argv
