from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class DatasetProtocol(Protocol):
    user_ids: Any
    timestamps: Any
    features: dict[str, Any]
    metadata: dict[str, Any]


class DataConfig(BaseModel):
    file_path: str = Field(min_length=1)
    file_format: Literal["csv", "parquet"]
    column_mapping: dict[str, str]
    base_path: str | None = None

    _KNOWN_ENV_VARS: ClassVar[list[str]] = ["RL_HEALTH_DATA_PATH", "DATA_PATH"]

    def resolved_path(self) -> str:
        if self.base_path:
            return (Path(self.base_path) / self.file_path).as_posix()
        for var in self._KNOWN_ENV_VARS:
            value = os.environ.get(var)
            if value is not None:
                return (Path(value) / self.file_path).as_posix()
        raise FileNotFoundError(
            f"No base_path set and none of {self._KNOWN_ENV_VARS} are set. "
            f"Cannot resolve {self.file_path}"
        )
