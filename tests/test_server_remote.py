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

    def test_auth_token_from_env(self):
        with patch.dict(os.environ, {"MNEMON_TOKEN": "secret123"}):
            import importlib
            import mnemon.server_remote as sr
            importlib.reload(sr)
            assert sr.AUTH_TOKEN == "secret123"

    def test_no_auth_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MNEMON_TOKEN", None)
            import importlib
            import mnemon.server_remote as sr
            importlib.reload(sr)
            assert sr.AUTH_TOKEN == ""


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
