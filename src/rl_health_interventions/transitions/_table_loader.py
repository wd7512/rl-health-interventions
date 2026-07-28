from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TableLoader:
    """Reads JSON transition-table files from a directory.

    Responsibility: file I/O — discover ``.json`` files in *table_dir*,
    open and parse each one, and return their contents as Python dicts.
    Malformed JSON files are logged as warnings and silently skipped.
    """

    def load_tables(self, table_dir: Path) -> list[dict]:
        """Load all JSON files from *table_dir* and return parsed contents.

        Files are read in sorted (by path) order. Files that are not valid
        JSON are skipped with a warning.
        """
        json_files = sorted(table_dir.glob("*.json"))
        tables: list[dict] = []
        for json_path in json_files:
            data = self._read_json(json_path)
            if data is not None:
                tables.append(data)
        return tables

    @staticmethod
    def _read_json(json_path: Path) -> dict | None:
        try:
            with json_path.open(encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in %s, skipping", json_path)
            return None
