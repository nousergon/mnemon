"""Tests for the remote-aware dashboard loaders.

Verifies that each loader dispatches correctly based on the remote-URL
detection, routes through ``memory_*`` MCP tools in remote mode, and
preserves the fallback path for local-vault development.

Streamlit caching is bypassed in tests — each call re-runs the loader
body so patch targets fire every time.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

# Dashboard deps are in the ``[ui]`` extra. When running the core
# ``[dev]`` test matrix these imports would crash the whole module at
# collection time — skip cleanly instead so the rest of the suite stays
# green.
pytest.importorskip("streamlit")
pytest.importorskip("umap", reason="umap-learn ships with [ui]")


@pytest.fixture(autouse=True)
def no_streamlit_cache(monkeypatch):
    """Neutralize ``@st.cache_data`` / ``@st.cache_resource`` so each test
    observes the underlying function body without Streamlit's memoization
    short-circuiting the dispatch we're trying to test."""
    import streamlit as st

    def passthrough(*dargs, **dkwargs):
        # Support both bare ``@cache_data`` (no parens) and
        # ``@cache_data(ttl=N)`` (with parens) decoration styles.
        if dargs and callable(dargs[0]):
            return dargs[0]
        def decorator(fn):
            return fn
        return decorator

    monkeypatch.setattr(st, "cache_data", passthrough)
    monkeypatch.setattr(st, "cache_resource", passthrough)
    # Force a reimport so the decorators actually apply as no-ops.
    import importlib
    import mnemon.dashboard.loaders as loaders_mod
    importlib.reload(loaders_mod)
    yield loaders_mod
    importlib.reload(loaders_mod)


@pytest.fixture
def remote(monkeypatch):
    """Pin the dashboard into remote mode via env var."""
    monkeypatch.setenv("MNEMON_REMOTE_URL", "https://example.fly.dev/mcp")


@pytest.fixture
def local(monkeypatch, tmp_path):
    """Pin the dashboard into local mode — env unset, no remote_url file."""
    monkeypatch.delenv("MNEMON_REMOTE_URL", raising=False)
    # Redirect home to a tmp path that doesn't contain a remote_url file.
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)


class TestUseRemoteDetection:
    def test_detects_remote_via_env(self, remote, no_streamlit_cache):
        assert no_streamlit_cache._use_remote() is True

    def test_local_mode_when_unset(self, local, no_streamlit_cache):
        assert no_streamlit_cache._use_remote() is False


class TestLoadStatus:
    def test_remote_calls_memory_status_and_parses_json(self, remote, no_streamlit_cache):
        payload = {"total_documents": 3, "total_vectors": 2, "by_type": [],
                   "pinned": 0, "invalidated": 0, "vault_path": "/data/x"}
        with patch(
            "mnemon.dashboard.loaders._call_remote",
            return_value=json.dumps(payload),
        ) as mock_call:
            result = no_streamlit_cache.load_status()
        mock_call.assert_called_once_with("memory_status", {})
        assert result == payload

    def test_local_falls_back_to_store(self, local, no_streamlit_cache):
        with patch("mnemon.dashboard.loaders.get_store") as mock_store:
            mock_store.return_value.status.return_value = {"total_documents": 5}
            result = no_streamlit_cache.load_status()
        assert result == {"total_documents": 5}


class TestLoadTimeline:
    def test_remote_passes_args(self, remote, no_streamlit_cache):
        with patch(
            "mnemon.dashboard.loaders._call_remote",
            return_value="[]",
        ) as mock_call:
            result = no_streamlit_cache.load_timeline(limit=50, content_type="note")
        mock_call.assert_called_once_with(
            "memory_timeline", {"limit": 50, "content_type": "note"},
        )
        assert result == []


class TestLoadSearch:
    def test_remote_passes_args(self, remote, no_streamlit_cache):
        with patch(
            "mnemon.dashboard.loaders._call_remote",
            return_value=json.dumps([{"doc_id": 1, "title": "hit"}]),
        ) as mock_call:
            result = no_streamlit_cache.load_search("q", limit=5, content_type="decision")
        mock_call.assert_called_once_with(
            "memory_search",
            {"query": "q", "limit": 5, "content_type": "decision"},
        )
        assert len(result) == 1
        assert result[0]["title"] == "hit"


class TestLoadDocument:
    def test_remote_hit_returns_doc(self, remote, no_streamlit_cache):
        doc = {"id": 7, "title": "T", "content": "C"}
        with patch(
            "mnemon.dashboard.loaders._call_remote",
            return_value=json.dumps(doc),
        ):
            assert no_streamlit_cache.load_document(7) == doc

    def test_remote_miss_returns_none(self, remote, no_streamlit_cache):
        """memory_get returns ``{"error": "not_found", "id": N}`` on miss —
        the loader normalizes that to ``None`` for page-level truthiness."""
        with patch(
            "mnemon.dashboard.loaders._call_remote",
            return_value=json.dumps({"error": "not_found", "id": 999}),
        ):
            assert no_streamlit_cache.load_document(999) is None


class TestLoadVectors:
    def test_remote_empty_returns_none(self, remote, no_streamlit_cache):
        with patch(
            "mnemon.dashboard.loaders._call_remote",
            return_value=json.dumps({"count": 0, "dim": 384,
                                     "truncated": False, "items": []}),
        ):
            assert no_streamlit_cache.load_vectors() is None

    def test_remote_builds_3_tuple(self, remote, no_streamlit_cache):
        """The server bakes the vec_id → doc join into ``memory_export_vectors``
        so the loader doesn't need a second query — important for browser
        clients that have no SQLite access."""
        items = [
            {"doc_id": 1, "vec_id": "abc_0", "title": "T1",
             "content_type": "note", "confidence": 0.5,
             "created_at": "2026-01-01", "pinned": False,
             "vector": [0.1] * 384},
            {"doc_id": 2, "vec_id": "def_0", "title": "T2",
             "content_type": "decision", "confidence": 0.9,
             "created_at": "2026-01-02", "pinned": True,
             "vector": [0.2] * 384},
        ]
        with patch(
            "mnemon.dashboard.loaders._call_remote",
            return_value=json.dumps({"count": 2, "dim": 384,
                                     "truncated": False, "items": items}),
        ):
            result = no_streamlit_cache.load_vectors()
        assert result is not None
        vec_ids, vectors, doc_map = result
        assert vec_ids == ["abc_0", "def_0"]
        assert vectors.shape == (2, 384)
        assert vectors.dtype == np.float32
        assert doc_map["abc_0"]["title"] == "T1"
        assert doc_map["def_0"]["content_type"] == "decision"
