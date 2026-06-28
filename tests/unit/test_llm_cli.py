"""Tests for the OpenCode Zen LLM client."""

from __future__ import annotations

import json
import logging
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from rl_health_interventions.cli.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMClientError,
    chat_completion,
    chat_text,
)

logger = logging.getLogger(__name__)


def _make_response(body: dict) -> MagicMock:
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock.read.return_value = json.dumps(body).encode("utf-8")
    return mock


def test_chat_completion_success():
    resp_body = {
        "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
        "usage": {"total_tokens": 10},
    }
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)):
        result = chat_completion(
            [{"role": "user", "content": "Hello"}],
            api_key="***",
        )

    assert result["choices"][0]["message"]["content"] == "Hi!"


def test_chat_completion_uses_correct_headers():
    resp_body = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }
    mock_resp = _make_response(resp_body)
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        chat_completion(
            [{"role": "user", "content": "test"}],
            api_key="***",
        )

    req = mock_urlopen.call_args[0][0]
    assert isinstance(req, urllib.request.Request)
    assert req.get_header("X-api-key") == "***"
    assert req.get_header("Content-type") == "application/json"


def test_chat_completion_uses_default_model():
    resp_body = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)) as m:
        chat_completion([{"role": "user", "content": "test"}], api_key="***")

    req = m.call_args[0][0]
    body = json.loads(req.data)
    assert body["model"] == DEFAULT_MODEL


def test_chat_completion_custom_model_and_params():
    resp_body = {
        "choices": [{"message": {"role": "assistant", "content": "done"}}],
    }
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)) as m:
        chat_completion(
            [{"role": "user", "content": "test"}],
            model="deepseek-v4-flash-free",
            temperature=0.7,
            max_tokens=128,
            api_key="***",
        )

    req = m.call_args[0][0]
    body = json.loads(req.data)
    assert body["model"] == "deepseek-v4-flash-free"
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 128


def test_chat_completion_http_error():
    from urllib.error import HTTPError

    error_resp = MagicMock()
    error_resp.read.return_value = (
        b'{"type":"error","error":{"type":"AuthError","message":"bad key"}}'
    )
    http_error = HTTPError(
        url="https://test", code=401, msg="Unauthorized", hdrs=MagicMock(), fp=None
    )
    # Manually assign .read to the error object
    http_error.read = error_resp.read

    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(LLMClientError, match="API error 401"):
            chat_completion([{"role": "user", "content": "test"}], api_key="***")


def test_chat_completion_network_error():
    from urllib.error import URLError

    with patch("urllib.request.urlopen", side_effect=URLError("Connection refused")):
        with pytest.raises(LLMClientError, match="Network error"):
            chat_completion([{"role": "user", "content": "test"}], api_key="***")


def test_chat_completion_no_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(LLMClientError, match="OPENCODE_API_KEY not set"):
            chat_completion([{"role": "user", "content": "test"}])


def test_chat_text_convenience():
    resp_body = {
        "choices": [{"message": {"role": "assistant", "content": "Hello world!"}}],
    }
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)):
        result = chat_text("Say hi", api_key="***")

    assert result == "Hello world!"


def test_default_base_url():
    assert DEFAULT_BASE_URL == "https://opencode.ai/zen/v1"


DEFAULT_RESP = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
