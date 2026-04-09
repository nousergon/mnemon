"""Tests for remote HTTP server configuration and health endpoint."""

import os
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from mnemon.server_remote import create_app


@pytest.fixture
def client():
    """Create a test client with no auth."""
    import mnemon.server_remote as sr
    sr.AUTH_TOKEN = ""
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_client():
    """Create a test client with bearer auth."""
    import mnemon.server_remote as sr
    sr.AUTH_TOKEN = "test-secret"
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


class TestHealth:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_bypasses_auth(self, auth_client):
        response = auth_client.get("/health")
        assert response.status_code == 200


class TestAuth:
    def test_mcp_rejects_without_token(self, auth_client):
        response = auth_client.post("/mcp", json={})
        assert response.status_code == 401

    def test_mcp_rejects_wrong_token(self, auth_client):
        response = auth_client.post(
            "/mcp",
            headers={"Authorization": "Bearer wrong-token"},
            json={},
        )
        assert response.status_code == 401

    def test_mcp_accepts_correct_token(self, auth_client):
        # With correct token, request passes auth. The MCP handler will
        # fail with 500 (task group not initialized in TestClient), but
        # that's not a 401 — auth passed.
        response = auth_client.post(
            "/mcp",
            headers={
                "Authorization": "Bearer test-secret",
                "Content-Type": "application/json",
            },
            json={"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.1.0"},
            }},
        )
        assert response.status_code != 401


class TestConfig:
    def test_default_port(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PORT", None)
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


class TestMcpServer:
    def test_mcp_has_13_tools(self):
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
