from __future__ import annotations

import json
import logging
import logging.handlers

from typing_extensions import override


class JsonFormatter(logging.Formatter):
    @override
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "asctime": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_data, default=str)


def setup_file_handler(path: str) -> logging.handlers.WatchedFileHandler:
    handler = logging.handlers.WatchedFileHandler(path)
    handler.setFormatter(JsonFormatter())
    return handler
