"""Tests for remote HTTP server configuration."""

import os
from unittest.mock import patch

import pytest


class TestRemoteConfig:
    def test_default_port(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PORT", None)
            # Re-import to pick up env
            import importlib
            import mnemon.server_remote as sr
            importlib.reload(sr)
            assert sr.PORT == 8502

    def test_custom_port(self):
        with patch.dict(os.environ, {"PORT": "9000"}):
            import importlib
            import mnemon.server_remote as sr
            importlib.reload(sr)
            assert sr.PORT == 9000

    def test_as_config_enabled_from_env(self, tmp_path):
        env = {
            "MNEMON_AS_ENABLED": "true",
            "MNEMON_PUBLIC_URL": "https://mnemon.example.com",
            "MNEMON_AS_PASSPHRASE": "x",
            "MNEMON_AS_KEY_DIR": str(tmp_path),
        }
        with patch.dict(os.environ, env):
            from mnemon.oauth_as import AuthorizationServerConfig
            config = AuthorizationServerConfig.from_env()
            assert config.enabled
            assert config.issuer == "https://mnemon.example.com"

    def test_as_config_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            for var in (
                "MNEMON_AS_ENABLED",
                "MNEMON_AS_PASSPHRASE",
                "MNEMON_PUBLIC_URL",
            ):
                os.environ.pop(var, None)
            from mnemon.oauth_as import AuthorizationServerConfig
            config = AuthorizationServerConfig.from_env()
            assert not config.enabled


class TestSessionManagerConfig:
    """Regression tests for the StreamableHTTP session manager wiring.

    These pin ``json_response=True`` because flipping it back to False
    re-introduces a hang: upstream's ``_session_creation_lock`` is held
    for the full duration of ``handle_request``, and in SSE response
    mode ``handle_request`` keeps the per-session SSE stream open until
    the client disconnects — so once one session is alive, every
    fresh-session POST queues behind the lock indefinitely. mnemon's
    tools are all single-shot RPCs, so json_response=True is the
    correct mode and must stay pinned True.
    """

    def test_session_manager_uses_json_response(self, monkeypatch, tmp_path):
        """Capture the kwargs passed to PersistentSessionManager when
        ``server_remote.main`` runs and assert ``json_response=True``.
        Bails out via SystemExit after the manager is instantiated so
        we don't actually start uvicorn."""
        import sys

        captured: dict = {}

        def fake_manager(*args, **kwargs):
            captured.update(kwargs)
            raise SystemExit("captured — abort before uvicorn.run")

        monkeypatch.setattr(
            "mnemon.persistent_sessions.PersistentSessionManager", fake_manager
        )
        monkeypatch.setenv("MNEMON_LOCAL_TOKEN", "x" * 32)
        monkeypatch.setenv("MNEMON_VAULT_DIR", str(tmp_path))
        monkeypatch.setenv("MNEMON_PUBLIC_URL", "http://127.0.0.1:8502")
        monkeypatch.setenv("MNEMON_ALLOWED_HOSTS", "127.0.0.1,127.0.0.1:8502")
        monkeypatch.delenv("MNEMON_AS_ENABLED", raising=False)

        for mod_name in ("mnemon.server_remote",):
            sys.modules.pop(mod_name, None)
        from mnemon.server_remote import run_remote

        with pytest.raises(SystemExit):
            run_remote()

        assert captured.get("json_response") is True, (
            f"PersistentSessionManager must be wired with json_response=True; "
            f"got {captured.get('json_response')!r}. See class docstring for why."
        )


class TestMcpServer:
    def test_mcp_has_tools(self):
        from mnemon.server import mcp
        tools = mcp._tool_manager._tools
        assert len(tools) == 14

    def test_mcp_tool_names(self):
        from mnemon.server import mcp
        tool_names = set(mcp._tool_manager._tools.keys())
        expected = {
            "memory_search",
            "memory_get", "memory_timeline",
            "memory_save", "memory_pin", "memory_forget",
            "memory_status", "memory_sweep", "memory_related",
            "memory_export_vectors",
            "memory_rebuild", "memory_check_contradictions",
            "profile_get", "profile_update",
        }
        assert tool_names == expected
