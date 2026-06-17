from __future__ import annotations

import sys

from rl_health_interventions.__main__ import main


def test_main_callable() -> None:
    assert callable(main)


def test_main_runs_with_default_config(tmp_path) -> None:
    """Calling main() without --config uses default and succeeds."""
    old_argv = sys.argv
    output_file = tmp_path / "results.csv"
    try:
        sys.argv = ["rl-health-interventions", "--output", str(output_file)]
        main()
    finally:
        sys.argv = old_argv
