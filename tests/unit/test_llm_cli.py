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
            api_key="test-key",
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
            api_key="test-key",
        )

    req = mock_urlopen.call_args[0][0]
    assert isinstance(req, urllib.request.Request)
    assert req.get_header("X-api-key") == "test-key"
    assert req.get_header("Content-type") == "application/json"
    assert req.get_header("User-agent") == "rl-health-interventions/0.1.0"


def test_chat_completion_uses_default_model():
    resp_body = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)) as m:
        chat_completion([{"role": "user", "content": "test"}], api_key="test-key")

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
            api_key="test-key",
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
    http_error.read = error_resp.read

    with (
        patch("urllib.request.urlopen", side_effect=http_error),
        pytest.raises(LLMClientError, match="API error 401"),
    ):
        chat_completion([{"role": "user", "content": "test"}], api_key="test-key")


def test_chat_completion_network_error():
    from urllib.error import URLError

    with (
        patch("urllib.request.urlopen", side_effect=URLError("Connection refused")),
        pytest.raises(LLMClientError, match="Network error"),
    ):
        chat_completion([{"role": "user", "content": "test"}], api_key="test-key")


def test_chat_completion_no_api_key():
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(LLMClientError, match="OPENCODE_API_KEY not set"),
    ):
        chat_completion([{"role": "user", "content": "test"}])


def test_chat_text_convenience():
    resp_body = {
        "choices": [{"message": {"role": "assistant", "content": "Hello world!"}}],
    }
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)):
        result = chat_text("Say hi", api_key="test-key")

    assert result == "Hello world!"


def test_chat_text_malformed_response_missing_choices():
    """chat_text raises LLMClientError when response has no 'choices' key."""
    resp_body = {"error": "something went wrong"}
    with (
        patch("urllib.request.urlopen", return_value=_make_response(resp_body)),
        pytest.raises(LLMClientError, match="Failed to parse API response"),
    ):
        chat_text("test", api_key="test-key")


def test_chat_text_malformed_response_empty_choices():
    """chat_text raises LLMClientError when choices list is empty."""
    resp_body = {"choices": []}
    with (
        patch("urllib.request.urlopen", return_value=_make_response(resp_body)),
        pytest.raises(LLMClientError, match="Failed to parse API response"),
    ):
        chat_text("test", api_key="test-key")


def test_chat_text_malformed_response_missing_message():
    """chat_text raises LLMClientError when message key is missing."""
    resp_body = {"choices": [{"finish_reason": "stop"}]}
    with (
        patch("urllib.request.urlopen", return_value=_make_response(resp_body)),
        pytest.raises(LLMClientError, match="Failed to parse API response"),
    ):
        chat_text("test", api_key="test-key")


def test_default_base_url():
    assert DEFAULT_BASE_URL == "https://opencode.ai/zen/v1"


def test_init_reexports():
    """cli.__init__ should re-export symbols from cli.llm, not duplicate them."""
    from rl_health_interventions import cli as init_cli
    from rl_health_interventions.cli import llm

    assert init_cli.DEFAULT_BASE_URL is llm.DEFAULT_BASE_URL
    assert init_cli.DEFAULT_MODEL is llm.DEFAULT_MODEL
    assert init_cli.LLMClientError is llm.LLMClientError
    assert init_cli.chat_completion is llm.chat_completion
    assert init_cli.chat_text is llm.chat_text
    assert init_cli.get_api_key is llm.get_api_key
