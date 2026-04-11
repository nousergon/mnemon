"""Tests for the remote MCP client helper used by mnemon hooks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from mnemon.hooks import _remote_client
from mnemon.hooks._remote_client import (
    RemoteClientConfigError,
    call_tool_sync,
    get_local_token,
    get_remote_url,
)


# ── get_remote_url ───────────────────────────────────────────────────────────


class TestGetRemoteUrl:
    def test_env_var_returned(self, monkeypatch):
        monkeypatch.setenv("MNEMON_REMOTE_URL", "https://example.fly.dev/mcp")
        assert get_remote_url() == "https://example.fly.dev/mcp"

    def test_env_var_stripped(self, monkeypatch):
        """Leading/trailing whitespace in the env var is ignored."""
        monkeypatch.setenv("MNEMON_REMOTE_URL", "  https://example.fly.dev/mcp  ")
        assert get_remote_url() == "https://example.fly.dev/mcp"

    def test_file_fallback(self, monkeypatch, tmp_path: Path):
        """When env is unset, the file at ~/.mnemon/remote_url is read."""
        monkeypatch.delenv("MNEMON_REMOTE_URL", raising=False)
        fake_dir = tmp_path / ".mnemon"
        fake_dir.mkdir()
        fake_file = fake_dir / "remote_url"
        fake_file.write_text("https://from-file.fly.dev/mcp\n")

        monkeypatch.setattr(_remote_client, "REMOTE_URL_FILE", fake_file)
        assert get_remote_url() == "https://from-file.fly.dev/mcp"

    def test_env_takes_precedence_over_file(self, monkeypatch, tmp_path: Path):
        """If both are set, env var wins — prevents surprises when a stale
        file exists alongside an explicit override."""
        fake_file = tmp_path / "remote_url"
        fake_file.write_text("https://from-file.fly.dev/mcp")
        monkeypatch.setattr(_remote_client, "REMOTE_URL_FILE", fake_file)

        monkeypatch.setenv("MNEMON_REMOTE_URL", "https://from-env.fly.dev/mcp")
        assert get_remote_url() == "https://from-env.fly.dev/mcp"

    def test_missing_raises_clear_error(self, monkeypatch, tmp_path: Path):
        """Both env and file unset should raise with actionable guidance."""
        monkeypatch.delenv("MNEMON_REMOTE_URL", raising=False)
        monkeypatch.setattr(
            _remote_client, "REMOTE_URL_FILE", tmp_path / "does-not-exist"
        )

        with pytest.raises(RemoteClientConfigError) as exc:
            get_remote_url()
        msg = str(exc.value)
        assert "MNEMON_REMOTE_URL" in msg
        assert "not configured" in msg
        # Error should surface the actual file path so the user knows
        # where to put the value.
        assert str(tmp_path / "does-not-exist") in msg

    def test_empty_file_treated_as_missing(self, monkeypatch, tmp_path: Path):
        """An empty file should not be silently accepted — raise instead."""
        monkeypatch.delenv("MNEMON_REMOTE_URL", raising=False)
        fake_file = tmp_path / "remote_url"
        fake_file.write_text("")
        monkeypatch.setattr(_remote_client, "REMOTE_URL_FILE", fake_file)

        with pytest.raises(RemoteClientConfigError):
            get_remote_url()


# ── get_local_token ──────────────────────────────────────────────────────────


class TestGetLocalToken:
    def test_env_var_returned(self, monkeypatch):
        monkeypatch.setenv("MNEMON_LOCAL_TOKEN", "env-token-abc")
        assert get_local_token() == "env-token-abc"

    def test_file_fallback(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv("MNEMON_LOCAL_TOKEN", raising=False)
        fake_file = tmp_path / "local_token"
        fake_file.write_text("file-token-xyz")
        monkeypatch.setattr(_remote_client, "LOCAL_TOKEN_FILE", fake_file)

        assert get_local_token() == "file-token-xyz"

    def test_file_trailing_newline_stripped(self, monkeypatch, tmp_path: Path):
        """A common footgun — trailing newline from ``echo`` would break
        hmac.compare_digest. The helper must strip it."""
        monkeypatch.delenv("MNEMON_LOCAL_TOKEN", raising=False)
        fake_file = tmp_path / "local_token"
        fake_file.write_text("file-token-xyz\n")
        monkeypatch.setattr(_remote_client, "LOCAL_TOKEN_FILE", fake_file)

        assert get_local_token() == "file-token-xyz"

    def test_env_takes_precedence_over_file(self, monkeypatch, tmp_path: Path):
        fake_file = tmp_path / "local_token"
        fake_file.write_text("file-token")
        monkeypatch.setattr(_remote_client, "LOCAL_TOKEN_FILE", fake_file)

        monkeypatch.setenv("MNEMON_LOCAL_TOKEN", "env-token")
        assert get_local_token() == "env-token"

    def test_missing_raises_clear_error(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv("MNEMON_LOCAL_TOKEN", raising=False)
        monkeypatch.setattr(
            _remote_client, "LOCAL_TOKEN_FILE", tmp_path / "does-not-exist"
        )

        with pytest.raises(RemoteClientConfigError) as exc:
            get_local_token()
        msg = str(exc.value)
        assert "MNEMON_LOCAL_TOKEN" in msg
        assert "not configured" in msg
        assert "chmod 600" in msg
        assert str(tmp_path / "does-not-exist") in msg


# ── call_tool_sync ───────────────────────────────────────────────────────────


class TestCallToolSync:
    """Exercises the sync wrapper. ``_call_tool_async`` is patched at the
    module level rather than at the MCP SDK layer — unit tests shouldn't
    care how the MCP SDK is wired, they care that sync→async→text works.
    """

    def test_returns_text_from_async(self, monkeypatch):
        async def _fake(tool_name, arguments, *, timeout, client_label):
            assert tool_name == "memory_search"
            assert arguments == {"query": "test", "limit": 5}
            assert timeout == 2.0
            assert client_label == "claude-code"
            return "fake tool output"

        monkeypatch.setattr(_remote_client, "_call_tool_async", _fake)

        result = call_tool_sync("memory_search", {"query": "test", "limit": 5})
        assert result == "fake tool output"

    def test_passes_custom_timeout_and_label(self, monkeypatch):
        captured = {}

        async def _fake(tool_name, arguments, *, timeout, client_label):
            captured["timeout"] = timeout
            captured["client_label"] = client_label
            return ""

        monkeypatch.setattr(_remote_client, "_call_tool_async", _fake)

        call_tool_sync(
            "memory_save",
            {"title": "t", "content": "c"},
            timeout=7.5,
            client_label="test-harness",
        )
        assert captured == {"timeout": 7.5, "client_label": "test-harness"}

    def test_timeout_raises(self, monkeypatch):
        async def _fake(tool_name, arguments, *, timeout, client_label):
            raise asyncio.TimeoutError("took too long")

        monkeypatch.setattr(_remote_client, "_call_tool_async", _fake)

        with pytest.raises(asyncio.TimeoutError):
            call_tool_sync("memory_search", {"query": "test"})

    def test_config_error_propagates(self, monkeypatch):
        async def _fake(tool_name, arguments, *, timeout, client_label):
            raise RemoteClientConfigError("nothing configured")

        monkeypatch.setattr(_remote_client, "_call_tool_async", _fake)

        with pytest.raises(RemoteClientConfigError):
            call_tool_sync("memory_search", {"query": "test"})

    def test_generic_exception_propagates(self, monkeypatch):
        """The sync wrapper must not swallow exceptions — hooks decide
        what to do with them."""
        async def _fake(tool_name, arguments, *, timeout, client_label):
            raise ConnectionError("econnrefused")

        monkeypatch.setattr(_remote_client, "_call_tool_async", _fake)

        with pytest.raises(ConnectionError):
            call_tool_sync("memory_search", {"query": "test"})
