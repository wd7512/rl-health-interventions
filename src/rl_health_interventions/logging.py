from __future__ import annotations

import json
import logging
import logging.handlers


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "asctime": self.formatTime(record),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
            },
            default=str,
        )


def setup_file_handler(path: str) -> logging.handlers.WatchedFileHandler:
    handler = logging.handlers.WatchedFileHandler(path)
    handler.setFormatter(JsonFormatter())
    return handler
