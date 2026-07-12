"""Generate cross-persona config files for all 7 table sets."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

BASE = Path("docs/experiments/sprint1_bootstrap")
CONFIGS = BASE / "configs"
CROSS = CONFIGS / "cross"

TABLE_SETS = [
    ("persona_base", "tables/persona/base_deepseek-v4-flash"),
    ("persona_goal_driven", "tables/persona/goal_driven_deepseek-v4-flash"),
    ("persona_resistant", "tables/persona/resistant_deepseek-v4-flash"),
    ("persona_social_responder", "tables/persona/social_responder_deepseek-v4-flash"),
    ("persona_stable_maintainer", "tables/persona/stable_maintainer_deepseek-v4-flash"),
    ("initial_deepseek", "tables/initial/deepseek"),
    ("initial_glm5.2", "tables/initial/glm5.2"),
]

TEMPLATES = [
    ("sprint1_bootstrap_extensions.yaml", "extensions"),
    ("sprint1_bootstrap_extensions_masked.yaml", "extensions_masked"),
    ("sprint1_bootstrap_context_burden.yaml", "context_burden"),
]


def rel_table_dir(config_dir: Path, table_path: str) -> str:
    """Compute relative path from config_dir to table_path (both relative to repo root)."""
    repo_root = Path.cwd()
    config_abs = (repo_root / config_dir).resolve()
    table_abs = (repo_root / table_path).resolve()
    return os.path.relpath(str(table_abs), str(config_abs))


def main() -> None:
    for short, table_path in TABLE_SETS:
        target_dir = CROSS / short
        target_dir.mkdir(parents=True, exist_ok=True)
        table_dir_rel = rel_table_dir(target_dir, table_path)

        for template_name, out_name in TEMPLATES:
            src = CONFIGS / template_name
            dst = target_dir / f"sprint1_bootstrap_{out_name}.yaml"
            content = src.read_text()
            new_line = f"table_dir: {table_dir_rel}"
            content = re.sub(r"^table_dir: .*$", new_line, content, flags=re.MULTILINE)
            dst.write_text(content)
            logging.info("  Wrote %s (table_dir: %s)", dst, table_dir_rel)

    logging.info("\nGenerated configs in %s", CROSS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
