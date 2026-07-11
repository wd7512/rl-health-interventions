"""Generate cross-persona config files for all 7 table sets."""

from __future__ import annotations
import os
from pathlib import Path

BASE = Path("docs/experimental_phases/sprint1_bootstrap")
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
            old_line = f"table_dir: ../../../../tables/persona/base_deepseek-v4-flash"
            new_line = f"table_dir: {table_dir_rel}"
            content = content.replace(old_line, new_line)
            # Also catch any other old table_dir variants (e.g. for base also used in initial)
            old_line2 = f"table_dir: ../../../../tables/initial/deepseek"
            content = content.replace(old_line2, new_line)
            dst.write_text(content)
            print(f"  Wrote {dst} (table_dir: {table_dir_rel})")

    print(f"\nGenerated configs in {CROSS}")


if __name__ == "__main__":
    main()
