"""Tests for the hook MemoryClient abstraction (:mod:`mnemon.hooks._client`).

Covers: ``has_remote_config`` config-source resolution, ``get_client``
picks the right client, ``RemoteMemoryClient`` delegates through the
real ``_remote_client`` module (so existing patches still fire), and
``LocalMemoryClient`` dispatches via ``mnemon.api.dispatch``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mnemon.hooks import _client
from mnemon.hooks._client import (
    LocalMemoryClient,
    MemoryClient,
    RemoteClientConfigError,
    RemoteMemoryClient,
    get_client,
    has_remote_config,
)


class TestHasRemoteConfig:
    def test_env_var_overrides_everything(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MNEMON_REMOTE_URL", "https://x.example/mcp")
        monkeypatch.setattr(
            _client, "REMOTE_URL_FILE", tmp_path / "remote_url"
        )
        assert has_remote_config() is True

    def test_file_when_env_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MNEMON_REMOTE_URL", raising=False)
        url_file = tmp_path / "remote_url"
        url_file.write_text("https://file.example/mcp\n")
        monkeypatch.setattr(_client, "REMOTE_URL_FILE", url_file)
        assert has_remote_config() is True

    def test_empty_file_is_falsy(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MNEMON_REMOTE_URL", raising=False)
        url_file = tmp_path / "remote_url"
        url_file.write_text("   \n")
        monkeypatch.setattr(_client, "REMOTE_URL_FILE", url_file)
        assert has_remote_config() is False

    def test_no_config_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MNEMON_REMOTE_URL", raising=False)
        monkeypatch.setattr(
            _client, "REMOTE_URL_FILE", tmp_path / "does-not-exist"
        )
        assert has_remote_config() is False


class TestGetClient:
    def test_returns_remote_when_configured(self, monkeypatch):
        monkeypatch.setenv("MNEMON_REMOTE_URL", "https://x/mcp")
        client = get_client()
        assert isinstance(client, RemoteMemoryClient)

    def test_returns_local_by_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MNEMON_REMOTE_URL", raising=False)
        monkeypatch.setattr(
            _client, "REMOTE_URL_FILE", tmp_path / "nope"
        )
        client = get_client()
        assert isinstance(client, LocalMemoryClient)

    def test_both_implement_protocol(self):
        assert isinstance(RemoteMemoryClient(), MemoryClient)
        assert isinstance(LocalMemoryClient(), MemoryClient)


class TestRemoteMemoryClient:
    def test_delegates_through_module_so_patches_fire(self):
        """Patches at ``mnemon.hooks._remote_client.call_tool_sync`` must
        still intercept remote calls routed through this wrapper. That
        invariant protects every existing hook test."""
        with patch(
            "mnemon.hooks._remote_client.call_tool_sync",
            return_value=("ok", 0.123),
        ) as mock_call:
            out, elapsed = RemoteMemoryClient().call_tool(
                "memory_status", {}, timeout=5.0, client_label="test"
            )
        assert (out, elapsed) == ("ok", 0.123)
        mock_call.assert_called_once_with(
            "memory_status", {}, timeout=5.0, client_label="test"
        )

    def test_propagates_config_error(self):
        with patch(
            "mnemon.hooks._remote_client.call_tool_sync",
            side_effect=RemoteClientConfigError("no url"),
        ):
            with pytest.raises(RemoteClientConfigError, match="no url"):
                RemoteMemoryClient().call_tool("memory_status", {})


class TestLocalMemoryClient:
    def test_dispatches_to_api_by_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMON_VAULT_DIR", str(tmp_path / "v"))
        # Reset the module-level default store so our isolated vault is used.
        import mnemon.api as api_mod

        api_mod._default_store = None
        out, elapsed = LocalMemoryClient().call_tool("memory_status", {})
        import json

        data = json.loads(out)
        assert "total_documents" in data
        assert elapsed >= 0.0

    def test_raises_unsupported_tool(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMON_VAULT_DIR", str(tmp_path / "v"))
        import mnemon.api as api_mod

        api_mod._default_store = None
        from mnemon.api import UnsupportedToolError

        with pytest.raises(UnsupportedToolError):
            LocalMemoryClient().call_tool("memory_not_a_thing", {})
