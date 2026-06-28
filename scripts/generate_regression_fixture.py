#!/usr/bin/env python
"""One-shot generator for the regression fixture.

Run from the project root:
    uv run python scripts/generate_regression_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rl_health_interventions.sweep import run_experiment

FIXTURE = Path("tests/fixtures/mvp_expected_rewards.json")
CONFIG = "config/rule_based.yaml"


def main() -> None:
    results = run_experiment(CONFIG)
    FIXTURE.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote fixture to {FIXTURE}")
    print(f"Contents: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
