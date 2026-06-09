from __future__ import annotations

import json
import logging
import logging.handlers
import tempfile
from pathlib import Path

from rl_health_interventions.logging import JsonFormatter, setup_file_handler


def test_json_formatter_produces_valid_json() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello world",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"
    assert parsed["name"] == "test"


def test_json_formatter_with_extra_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="episode %d done",
        args=(42,),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "episode 42 done"


def test_setup_file_handler_returns_watched_file_handler() -> None:
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        path = f.name
    handler = setup_file_handler(path)
    assert isinstance(handler, logging.handlers.WatchedFileHandler)
    assert isinstance(handler.formatter, JsonFormatter)
    handler.close()
    Path(path).unlink()
