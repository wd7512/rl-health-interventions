from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Registry:
    """Generic registry with import-time module registration."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, type] = {}

    def register(self, key: str, cls: type) -> None:
        self._entries[key] = cls

    def make(self, name: str, **kwargs: Any) -> Any:
        if name not in self._entries:
            msg = f"Unknown {self._name}: {name}. Known: {list(self._entries)}"
            raise KeyError(msg)
        return self._entries[name](**kwargs)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __getitem__(self, key: str) -> type:
        return self._entries[key]

    def keys(self) -> Any:
        return self._entries.keys()

    def items(self) -> Any:
        return self._entries.items()

    def __iter__(self) -> Any:
        return iter(self._entries)

    def load_modules(self, modules: list, logger_name: str | None = None) -> None:
        """Call register() on each module, logging failures."""
        log = logging.getLogger(logger_name or __name__)
        errors: list[str] = []
        for mod in modules:
            try:
                mod.register()
            except Exception as exc:
                log.exception("Failed to register %s", mod.__name__)
                errors.append(f"{mod.__name__}: {exc}")
        if errors:
            msg = f"Failed to register {len(errors)} module(s): {'; '.join(errors)}"
            raise RuntimeError(msg)
