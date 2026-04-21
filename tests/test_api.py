"""Tests for the in-process tool surface (:mod:`mnemon.api`).

Every handler must return the same shape as the matching
``@mcp.tool()`` in :mod:`mnemon.server` so hooks dispatching locally
consume identical JSON. Tests exercise a real :class:`Store` on a temp
vault (SQLite is fast; no mocking needed for storage), and stub the
embedder to keep the test runtime independent of FastEmbed.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from mnemon import api
from mnemon.store import Store


@pytest.fixture
def store(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    return Store(db_path=str(vault_dir / "default.sqlite"))


class TestMemorySave:
    def test_returns_server_shape_message(self, store):
        with patch(
            "mnemon.embedder.embed_document", return_value=None, create=False
        ):
            out = api.memory_save(
                "Test title",
                "Test content body",
                content_type="observation",
                store=store,
            )
        # Format mirrors server.memory_save exactly so downstream parsers
        # don't have to branch on transport.
        assert out.startswith("Saved memory #")
        assert '"Test title"' in out
        assert "[observation]" in out

    def test_best_effort_embedding_failure_does_not_raise(self, store):
        with patch(
            "mnemon.embedder.embed_document",
            side_effect=RuntimeError("boom"),
            create=False,
        ):
            out = api.memory_save(
                "Title",
                "Content",
                content_type="note",
                store=store,
            )
        assert out.startswith("Saved memory #")


class TestMemorySearch:
    def test_empty_vault_returns_json_empty_list(self, store):
        raw = api.memory_search("anything", limit=5, store=store)
        assert json.loads(raw) == []

    def test_fields_match_server_schema(self, store):
        with patch(
            "mnemon.embedder.embed_document", return_value=None, create=False
        ):
            api.memory_save(
                "Alpha engine paper trading notes",
                "Running Alpha Engine on IB paper account with 15-min delay",
                content_type="note",
                store=store,
            )
        raw = api.memory_search("alpha engine", limit=10, store=store)
        results = json.loads(raw)
        assert len(results) >= 1
        hit = results[0]
        # Exact fields server.memory_search emits
        expected_keys = {
            "doc_id",
            "title",
            "content",
            "content_type",
            "confidence",
            "composite_score",
            "vector_similarity",
            "created_at",
        }
        assert expected_keys == set(hit)


class TestMemoryStatus:
    def test_returns_json_with_vault_path(self, store):
        raw = api.memory_status(store=store)
        data = json.loads(raw)
        # Schema from store.status() — downstream callers (dashboard, doctor)
        # depend on these keys.
        assert "total_documents" in data
        assert "vault_path" in data


class TestMemoryGet:
    def test_hit_returns_full_document(self, store):
        with patch(
            "mnemon.embedder.embed_document", return_value=None, create=False
        ):
            out = api.memory_save(
                "Findable", "body", content_type="note", store=store
            )
        doc_id = int(out.split("#")[1].split(":")[0])

        raw = api.memory_get(doc_id, store=store)
        data = json.loads(raw)
        assert data["id"] == doc_id
        assert data["title"] == "Findable"

    def test_miss_returns_error_shape(self, store):
        raw = api.memory_get(999_999, store=store)
        data = json.loads(raw)
        assert data == {"error": "not_found", "id": 999_999}


class TestMemoryForget:
    def test_forgets_and_reports(self, store):
        with patch(
            "mnemon.embedder.embed_document", return_value=None, create=False
        ):
            out = api.memory_save("T", "C", store=store)
        doc_id = int(out.split("#")[1].split(":")[0])
        msg = api.memory_forget(doc_id, store=store)
        assert f"Forgot memory #{doc_id}" in msg

    def test_forget_missing_reports_miss(self, store):
        msg = api.memory_forget(999_999, store=store)
        assert "not found" in msg.lower()


class TestDispatch:
    def test_routes_by_name(self, store):
        raw = api.dispatch("memory_status", {}, store=store)
        assert "total_documents" in json.loads(raw)

    def test_unsupported_tool_raises(self, store):
        with pytest.raises(api.UnsupportedToolError, match="memory_nope"):
            api.dispatch("memory_nope", {}, store=store)

    def test_unknown_kwargs_ignored(self, store):
        """FastMCP silently drops extra kwargs. Match that so callers
        can't accidentally break by passing a newer MCP arg we haven't
        plumbed through yet."""
        raw = api.dispatch(
            "memory_status",
            {"unused_field": "surprise", "another": 42},
            store=store,
        )
        assert "total_documents" in json.loads(raw)


class TestDefaultStoreLazy:
    def test_default_store_created_once(self, tmp_path, monkeypatch):
        # Redirect MNEMON_VAULT_DIR so we don't pollute the real vault.
        monkeypatch.setenv("MNEMON_VAULT_DIR", str(tmp_path / "default"))
        # Reset module-level cache between tests.
        import mnemon.api as api_mod

        api_mod._default_store = None
        with patch(
            "mnemon.embedder.embed_document", return_value=None, create=False
        ):
            api.memory_save("x", "y")
        first = api_mod._default_store
        assert first is not None
        api.memory_status()
        assert api_mod._default_store is first
