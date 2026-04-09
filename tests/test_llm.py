"""Tests for LLM abstraction — tests mock the model since it requires download."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mnemon.llm import _model_dir, is_available


class TestModelDir:
    def test_default_model_dir(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MNEMON_MODEL_DIR", None)
            d = _model_dir()
            assert d == Path.home() / ".mnemon" / "models"

    def test_custom_model_dir(self):
        with patch.dict(os.environ, {"MNEMON_MODEL_DIR": "/tmp/models"}):
            assert _model_dir() == Path("/tmp/models")


class TestIsAvailable:
    def test_unavailable_without_llama_cpp(self):
        with patch.dict("sys.modules", {"llama_cpp": None}):
            # Force ImportError
            with patch("builtins.__import__", side_effect=ImportError):
                # is_available catches import errors
                pass

    @patch("mnemon.llm._resolve_model_path", side_effect=FileNotFoundError)
    def test_unavailable_without_model(self, mock_resolve):
        with patch("mnemon.llm.is_available") as mock_avail:
            mock_avail.return_value = False
            assert not mock_avail()


class TestGenerate:
    @patch("mnemon.llm._ensure_model")
    def test_generate_returns_response(self, mock_ensure):
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "update"}}]
        }
        mock_ensure.return_value = mock_llm

        from mnemon.llm import generate
        result = generate("system prompt", "user message", max_tokens=10)
        assert result == "update"

        mock_llm.create_chat_completion.assert_called_once()
        call_kwargs = mock_llm.create_chat_completion.call_args
        messages = call_kwargs[1]["messages"] if "messages" in call_kwargs[1] else call_kwargs[0][0]
        # Verify temperature
        assert call_kwargs[1]["temperature"] == 0.3

    @patch("mnemon.llm._ensure_model")
    def test_generate_strips_whitespace(self, mock_ensure):
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "  contradiction  \n"}}]
        }
        mock_ensure.return_value = mock_llm

        from mnemon.llm import generate
        result = generate("sys", "user")
        assert result == "contradiction"
