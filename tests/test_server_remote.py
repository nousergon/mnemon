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

    def test_oauth_config_enabled_from_env(self):
        env = {
            "MNEMON_OAUTH_ISSUER": "https://issuer.example.com/",
            "MNEMON_OAUTH_JWKS_URL": "https://issuer.example.com/.well-known/jwks.json",
            "MNEMON_OAUTH_AUDIENCE": "https://mnemon.example.com/mcp",
        }
        with patch.dict(os.environ, env):
            from mnemon.auth import OAuthConfig
            config = OAuthConfig.from_env()
            assert config.enabled
            assert config.issuer == "https://issuer.example.com/"

    def test_oauth_config_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            for var in (
                "MNEMON_OAUTH_ISSUER",
                "MNEMON_OAUTH_JWKS_URL",
                "MNEMON_OAUTH_AUDIENCE",
            ):
                os.environ.pop(var, None)
            from mnemon.auth import OAuthConfig
            config = OAuthConfig.from_env()
            assert not config.enabled


class TestMcpServer:
    def test_mcp_has_tools(self):
        from mnemon.server import mcp
        tools = mcp._tool_manager._tools
        assert len(tools) == 13

    def test_mcp_tool_names(self):
        from mnemon.server import mcp
        tool_names = set(mcp._tool_manager._tools.keys())
        expected = {
            "memory_search", "memory_get", "memory_timeline",
            "memory_save", "memory_pin", "memory_forget",
            "memory_status", "memory_sweep", "memory_related",
            "memory_rebuild", "memory_check_contradictions",
            "profile_get", "profile_update",
        }
        assert tool_names == expected
