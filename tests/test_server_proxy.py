"""Tests for the remote-proxy MCP server (``mnemon.server_proxy``).

The proxy must expose the EXACT same tool surface as the local
``mnemon.server`` (so a client can't tell which backend it's talking to)
while forwarding every call to the remote vault and never touching the
local SQLite store.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mnemon import server as local_server
from mnemon import server_proxy


def _tools(mcp):
    """name -> Tool, via the sync tool-manager listing."""
    return {t.name: t for t in mcp._tool_manager.list_tools()}


class TestToolParity:
    """The proxy can't drift from the real server — same names, schemas,
    and descriptions, derived structurally via functools.wraps."""

    def test_same_tool_names(self):
        proxy = server_proxy.build_proxy()
        assert set(_tools(proxy)) == set(_tools(local_server.mcp))

    def test_same_input_schemas(self):
        proxy_tools = _tools(server_proxy.build_proxy())
        local_tools = _tools(local_server.mcp)
        # ``.parameters`` is the JSON input schema on the tool-manager's
        # internal Tool objects (the async protocol listing calls it
        # ``inputSchema``).
        mismatches = {
            name
            for name, lt in local_tools.items()
            if proxy_tools[name].parameters != lt.parameters
        }
        assert not mismatches, f"schema drift in: {mismatches}"

    def test_same_descriptions(self):
        proxy_tools = _tools(server_proxy.build_proxy())
        local_tools = _tools(local_server.mcp)
        mismatches = {
            name
            for name, lt in local_tools.items()
            if (proxy_tools[name].description or "") != (lt.description or "")
        }
        assert not mismatches, f"description drift in: {mismatches}"


class TestForwarding:
    """Every tool body forwards to the remote via call_tool_sync and
    returns its text verbatim — the local Store is never constructed."""

    def test_forward_passes_args_and_drops_none(self):
        captured = {}

        def fake_call(tool_name, arguments, *, timeout, client_label):
            captured["tool_name"] = tool_name
            captured["arguments"] = arguments
            captured["timeout"] = timeout
            captured["client_label"] = client_label
            return ("REMOTE_RESULT", 0.01)

        with patch.object(server_proxy, "call_tool_sync", fake_call):
            proxy = server_proxy.build_proxy()
            fn = _tools(proxy)["memory_search"].fn
            out = fn(query="hello", limit=5, content_type=None)

        assert out == "REMOTE_RESULT"
        assert captured["tool_name"] == "memory_search"
        # None-valued optionals are dropped so the remote uses its default
        assert captured["arguments"] == {"query": "hello", "limit": 5}
        assert captured["client_label"] == "claude-code-proxy"
        assert captured["timeout"] == server_proxy.PROXY_TIMEOUT_SEC

    def test_forward_keeps_explicit_non_none_optionals(self):
        captured = {}

        def fake_call(tool_name, arguments, *, timeout, client_label):
            captured["arguments"] = arguments
            return ("ok", 0.0)

        with patch.object(server_proxy, "call_tool_sync", fake_call):
            proxy = server_proxy.build_proxy()
            fn = _tools(proxy)["memory_search"].fn
            fn(query="q", content_type="note")

        assert captured["arguments"] == {
            "query": "q",
            "limit": 10,
            "content_type": "note",
        }

    def test_forward_never_opens_local_store(self):
        """A proxy tool call must not construct a local Store."""
        with patch.object(
            server_proxy, "call_tool_sync", lambda *a, **k: ("ok", 0.0)
        ), patch("mnemon.store.Store") as store_cls:
            proxy = server_proxy.build_proxy()
            _tools(proxy)["memory_status"].fn()
            store_cls.assert_not_called()

    def test_failure_propagates_not_swallowed(self):
        """Fail-loud: a remote error raises out of the tool call rather
        than degrading to the local vault."""
        def boom(*a, **k):
            raise RuntimeError("remote unreachable")

        with patch.object(server_proxy, "call_tool_sync", boom):
            proxy = server_proxy.build_proxy()
            fn = _tools(proxy)["memory_status"].fn
            with pytest.raises(RuntimeError, match="remote unreachable"):
                fn()
