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


class TestMcpServer:
    def test_mcp_has_tools(self):
        from mnemon.server import mcp
        tools = mcp._tool_manager._tools
        assert len(tools) == 14

    def test_mcp_tool_names(self):
        from mnemon.server import mcp
        tool_names = set(mcp._tool_manager._tools.keys())
        expected = {
            "memory_search", "memory_search_structured",
            "memory_get", "memory_timeline",
            "memory_save", "memory_pin", "memory_forget",
            "memory_status", "memory_sweep", "memory_related",
            "memory_rebuild", "memory_check_contradictions",
            "profile_get", "profile_update",
        }
        assert tool_names == expected
