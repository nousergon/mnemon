"""Tests for remote HTTP server."""

import os
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from mnemon.server_remote import create_app


@pytest.fixture
def client():
    """Create a test client with no auth."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MNEMON_TOKEN", None)
        # Reimport to pick up env change
        import mnemon.server_remote as sr
        sr.AUTH_TOKEN = ""
        app = create_app()
        return TestClient(app)


@pytest.fixture
def auth_client():
    """Create a test client with bearer auth."""
    import mnemon.server_remote as sr
    sr.AUTH_TOKEN = "test-secret"
    app = create_app()
    return TestClient(app)


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
        response = auth_client.post("/mcp")
        assert response.status_code == 401

    def test_mcp_rejects_wrong_token(self, auth_client):
        response = auth_client.post(
            "/mcp",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401

    def test_mcp_accepts_correct_token(self, auth_client):
        # MCP endpoint expects proper JSON-RPC, so we'll get a protocol-level
        # response (not 401) which means auth passed
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
        # Should not be 401 — auth passed
        assert response.status_code != 401


class TestMcpEndpoint:
    def test_mcp_404_for_unknown_path(self, client):
        response = client.get("/unknown")
        assert response.status_code == 404
