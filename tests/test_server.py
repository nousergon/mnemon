"""Tests for the MCP server tool handlers."""

from unittest.mock import MagicMock, patch

import pytest

import mnemon.server as server_mod
from mnemon.server import (
    memory_check_contradictions,
    memory_forget,
    memory_get,
    memory_pin,
    memory_rebuild,
    memory_related,
    memory_save,
    memory_search,
    memory_status,
    memory_sweep,
    memory_timeline,
    profile_get,
    profile_update,
)


@pytest.fixture(autouse=True)
def reset_store():
    """Reset the singleton store between tests."""
    server_mod._store = None
    yield
    server_mod._store = None


def _make_search_result(**overrides):
    """Build a mock ScoredResult (as returned by search())."""
    defaults = {
        "doc_id": 1,
        "title": "Test Memory",
        "content": "Some content",
        "content_type": "note",
        "memory_type": "explicit",
        "confidence": 0.80,
        "created_at": "2026-04-01T00:00:00Z",
        "composite_score": 0.75,
    }
    defaults.update(overrides)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


def _make_document(**overrides):
    """Build a mock Document."""
    defaults = {
        "id": 1,
        "collection": "default",
        "path": None,
        "title": "Test Doc",
        "hash": "abc123",
        "content_type": "note",
        "memory_type": "explicit",
        "confidence": 0.80,
        "access_count": 1,
        "is_pinned": False,
        "is_invalidated": False,
        "created_at": "2026-04-01T00:00:00Z",
        "updated_at": "2026-04-01T00:00:00Z",
        "content": "Full document content here.",
        "source_client": None,
    }
    defaults.update(overrides)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


def _make_sweep_candidate(**overrides):
    """Build a mock SweepCandidate."""
    defaults = {
        "id": 5,
        "title": "Old Memory",
        "content_type": "note",
        "age_days": 120,
    }
    defaults.update(overrides)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


def _make_related(**overrides):
    """Build a mock RelatedDocument."""
    defaults = {
        "id": 2,
        "title": "Related Doc",
        "relation_type": "supports",
        "weight": 0.85,
    }
    defaults.update(overrides)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


# ── _get_store ───────────────────────────────────────────────────────────────


class TestGetStore:
    @patch("mnemon.server.Store")
    def test_creates_store_on_first_call(self, MockStore):
        store = server_mod._get_store()
        MockStore.assert_called_once()
        assert store is MockStore.return_value

    @patch("mnemon.server.Store")
    def test_returns_same_store_on_subsequent_calls(self, MockStore):
        s1 = server_mod._get_store()
        s2 = server_mod._get_store()
        MockStore.assert_called_once()
        assert s1 is s2


# ── memory_search ────────────────────────────────────────────────────────────


class TestMemorySearch:
    @patch("mnemon.server.search")
    @patch("mnemon.server.Store")
    def test_no_results(self, MockStore, mock_search):
        mock_search.return_value = []
        result = memory_search("test query")
        assert result == "No memories found matching your query."

    @patch("mnemon.server.search")
    @patch("mnemon.server.Store")
    def test_with_results(self, MockStore, mock_search):
        r1 = _make_search_result(doc_id=1, title="Alpha", composite_score=0.9, confidence=0.85)
        r2 = _make_search_result(doc_id=2, title="Beta", composite_score=0.7, confidence=0.60)
        mock_search.return_value = [r1, r2]

        result = memory_search("test query", limit=5)
        assert "1. [note] **Alpha**" in result
        assert "2. [note] **Beta**" in result
        assert "score: 0.900" in result
        assert "confidence: 0.85" in result
        assert "id: 1" in result
        assert "id: 2" in result

    @patch("mnemon.server.search")
    @patch("mnemon.server.Store")
    def test_long_content_truncated(self, MockStore, mock_search):
        long_content = "x" * 500
        r = _make_search_result(content=long_content)
        mock_search.return_value = [r]

        result = memory_search("query")
        assert "..." in result
        # Only first 300 chars of content should appear
        assert "x" * 300 in result
        assert "x" * 301 not in result

    @patch("mnemon.server.search")
    @patch("mnemon.server.Store")
    def test_short_content_no_ellipsis(self, MockStore, mock_search):
        r = _make_search_result(content="short")
        mock_search.return_value = [r]

        result = memory_search("query")
        assert "..." not in result

    @patch("mnemon.server.search")
    @patch("mnemon.server.Store")
    def test_passes_content_type(self, MockStore, mock_search):
        mock_search.return_value = []
        memory_search("query", content_type="decision")
        mock_search.assert_called_once_with(
            MockStore.return_value, "query", limit=10, content_type="decision"
        )


# ── memory_get ───────────────────────────────────────────────────────────────


class TestMemoryGet:
    @patch("mnemon.server.Store")
    def test_found(self, MockStore):
        doc = _make_document(
            title="My Decision",
            content_type="decision",
            confidence=0.90,
            created_at="2026-04-01",
            content="We chose option A.",
        )
        MockStore.return_value.get.return_value = doc

        result = memory_get(1)
        assert result.startswith("# My Decision")
        assert "decision" in result
        assert "0.90" in result
        assert "We chose option A." in result

    @patch("mnemon.server.Store")
    def test_not_found(self, MockStore):
        MockStore.return_value.get.return_value = None
        result = memory_get(999)
        assert result == "Memory #999 not found."


# ── memory_timeline ──────────────────────────────────────────────────────────


class TestMemoryTimeline:
    @patch("mnemon.server.Store")
    def test_empty(self, MockStore):
        MockStore.return_value.timeline.return_value = []
        result = memory_timeline()
        assert result == "No memories found."

    @patch("mnemon.server.Store")
    def test_populated(self, MockStore):
        d1 = _make_document(id=10, title="First", content_type="note", created_at="2026-04-01")
        d2 = _make_document(id=11, title="Second", content_type="decision", created_at="2026-04-02")
        MockStore.return_value.timeline.return_value = [d1, d2]

        result = memory_timeline(limit=5, content_type="note")
        assert "**First** [note]" in result
        assert "**Second** [decision]" in result
        assert "id: 10" in result
        assert "id: 11" in result

    @patch("mnemon.server.Store")
    def test_passes_args(self, MockStore):
        MockStore.return_value.timeline.return_value = []
        memory_timeline(limit=30, content_type="preference")
        MockStore.return_value.timeline.assert_called_once_with(30, "preference")


# ── memory_save ──────────────────────────────────────────────────────────────


class TestMemorySave:
    @patch("mnemon.server.Store")
    def test_success(self, MockStore):
        mock_store = MockStore.return_value
        mock_store.save.return_value = 42
        mock_store.get.return_value = _make_document(id=42, hash="h42")

        with patch("mnemon.server.embed_document", create=True):
            result = memory_save("My Note", "Content here", content_type="note")

        assert result == 'Saved memory #42: "My Note" [note]'
        mock_store.save.assert_called_once_with(
            title="My Note",
            content="Content here",
            content_type="note",
            collection="default",
            source_client=None,
        )

    @patch("mnemon.server.Store")
    def test_embedding_failure_still_succeeds(self, MockStore):
        mock_store = MockStore.return_value
        mock_store.save.return_value = 7
        mock_store.get.return_value = _make_document(id=7, hash="h7")

        # The embed import inside memory_save will raise — that's fine
        result = memory_save("Title", "Content")
        assert result == 'Saved memory #7: "Title" [note]'

    @patch("mnemon.server.Store")
    def test_passes_all_params(self, MockStore):
        mock_store = MockStore.return_value
        mock_store.save.return_value = 1
        mock_store.get.return_value = None  # embed won't run

        memory_save(
            "T", "C",
            content_type="decision",
            collection="work",
            source_client="claude",
        )
        mock_store.save.assert_called_once_with(
            title="T",
            content="C",
            content_type="decision",
            collection="work",
            source_client="claude",
        )


# ── memory_pin ───────────────────────────────────────────────────────────────


class TestMemoryPin:
    @patch("mnemon.server.Store")
    def test_success(self, MockStore):
        MockStore.return_value.pin.return_value = True
        result = memory_pin(5)
        assert result == "Pinned memory #5."

    @patch("mnemon.server.Store")
    def test_not_found(self, MockStore):
        MockStore.return_value.pin.return_value = False
        result = memory_pin(999)
        assert result == "Memory #999 not found."


# ── memory_forget ────────────────────────────────────────────────────────────


class TestMemoryForget:
    @patch("mnemon.server.Store")
    def test_success(self, MockStore):
        MockStore.return_value.forget.return_value = True
        result = memory_forget(3)
        assert result == "Forgot memory #3."

    @patch("mnemon.server.Store")
    def test_not_found(self, MockStore):
        MockStore.return_value.forget.return_value = False
        result = memory_forget(999)
        assert result == "Memory #999 not found or already forgotten."


# ── memory_status ────────────────────────────────────────────────────────────


class TestMemoryStatus:
    @patch("mnemon.server.Store")
    def test_returns_formatted_status(self, MockStore):
        MockStore.return_value.status.return_value = {
            "vault_path": "/home/user/.mnemon/default.sqlite",
            "total_documents": 42,
            "total_vectors": 38,
            "pinned": 3,
            "invalidated": 1,
            "by_type": [
                {"content_type": "note", "count": 30},
                {"content_type": "decision", "count": 12},
            ],
        }

        result = memory_status()
        assert "Vault: /home/user/.mnemon/default.sqlite" in result
        assert "Total memories: 42" in result
        assert "Vectors: 38" in result
        assert "Pinned: 3" in result
        assert "Invalidated: 1" in result
        assert "note: 30" in result
        assert "decision: 12" in result

    @patch("mnemon.server.Store")
    def test_empty_vault(self, MockStore):
        MockStore.return_value.status.return_value = {
            "vault_path": "/tmp/test.sqlite",
            "total_documents": 0,
            "total_vectors": 0,
            "pinned": 0,
            "invalidated": 0,
            "by_type": [],
        }

        result = memory_status()
        assert "Total memories: 0" in result
        assert "By type:" in result


# ── memory_sweep ─────────────────────────────────────────────────────────────


class TestMemorySweep:
    @patch("mnemon.server.Store")
    def test_no_candidates(self, MockStore):
        MockStore.return_value.sweep.return_value = {"candidates": []}
        result = memory_sweep()
        assert result == "No stale memories to archive."

    @patch("mnemon.server.Store")
    def test_dry_run_with_candidates(self, MockStore):
        c1 = _make_sweep_candidate(id=10, title="Old Note", content_type="note", age_days=90)
        c2 = _make_sweep_candidate(id=11, title="Stale Obs", content_type="observation", age_days=200)
        MockStore.return_value.sweep.return_value = {"candidates": [c1, c2]}

        result = memory_sweep(dry_run=True)
        assert "Would archive 2 memories:" in result
        assert '#10 "Old Note" [note]' in result
        assert '#11 "Stale Obs" [observation]' in result
        assert "90 days old" in result

    @patch("mnemon.server.Store")
    def test_real_sweep(self, MockStore):
        c = _make_sweep_candidate(id=5, title="Gone", content_type="note", age_days=365)
        MockStore.return_value.sweep.return_value = {"candidates": [c]}

        result = memory_sweep(dry_run=False)
        assert "Archived 1 memories:" in result
        assert "Would archive" not in result

    @patch("mnemon.server.Store")
    def test_passes_dry_run_arg(self, MockStore):
        MockStore.return_value.sweep.return_value = {"candidates": []}
        memory_sweep(dry_run=False)
        MockStore.return_value.sweep.assert_called_once_with(False)


# ── memory_related ───────────────────────────────────────────────────────────


class TestMemoryRelated:
    @patch("mnemon.server.Store")
    def test_empty(self, MockStore):
        MockStore.return_value.get_related.return_value = []
        result = memory_related(1)
        assert result == "No related memories found for #1."

    @patch("mnemon.server.Store")
    def test_populated(self, MockStore):
        r1 = _make_related(id=2, title="Supporting Doc", relation_type="supports", weight=0.90)
        r2 = _make_related(id=3, title="Contradicting Doc", relation_type="contradicts", weight=0.70)
        MockStore.return_value.get_related.return_value = [r1, r2]

        result = memory_related(1, limit=5)
        assert "[supports] **Supporting Doc**" in result
        assert "weight: 0.90" in result
        assert "[contradicts] **Contradicting Doc**" in result
        assert "id: 2" in result
        assert "id: 3" in result

    @patch("mnemon.server.Store")
    def test_passes_args(self, MockStore):
        MockStore.return_value.get_related.return_value = []
        memory_related(7, limit=3)
        MockStore.return_value.get_related.assert_called_once_with(7, 3)


# ── memory_rebuild ───────────────────────────────────────────────────────────


class TestMemoryRebuild:
    @patch("mnemon.server.Store")
    def test_success(self, MockStore):
        d1 = _make_document(hash="h1", title="Doc 1", content="Content 1")
        d2 = _make_document(hash="h2", title="Doc 2", content="Content 2")
        MockStore.return_value.timeline.return_value = [d1, d2]

        with patch.dict("sys.modules", {"mnemon.embedder": MagicMock()}):
            with patch("mnemon.server.embed_document", create=True) as mock_embed:
                # Need to patch the import inside the function
                import importlib
                # Instead, patch at the point of import within memory_rebuild
                pass

        # More direct approach: patch the import mechanism
        mock_embed_fn = MagicMock()
        with patch.dict(
            "sys.modules",
            {"mnemon.embedder": MagicMock(embed_document=mock_embed_fn)},
        ):
            result = memory_rebuild()

        assert "2 documents embedded" in result
        assert "0 failed" in result

    @patch("mnemon.server.Store")
    def test_import_error_no_fastembed(self, MockStore):
        MockStore.return_value.timeline.return_value = [_make_document()]

        # Ensure the import fails
        import sys
        original = sys.modules.get("mnemon.embedder")
        sys.modules["mnemon.embedder"] = None  # type: ignore[assignment]
        try:
            result = memory_rebuild()
        finally:
            if original is not None:
                sys.modules["mnemon.embedder"] = original
            else:
                sys.modules.pop("mnemon.embedder", None)

        assert "FastEmbed not installed" in result

    @patch("mnemon.server.Store")
    def test_partial_failure(self, MockStore):
        d1 = _make_document(hash="h1", title="Good", content="ok")
        d2 = _make_document(hash="h2", title="Bad", content="fail")
        d3 = _make_document(hash="h3", title="Good2", content="ok2")
        MockStore.return_value.timeline.return_value = [d1, d2, d3]

        mock_embed_fn = MagicMock(side_effect=[None, Exception("embed error"), None])
        with patch.dict(
            "sys.modules",
            {"mnemon.embedder": MagicMock(embed_document=mock_embed_fn)},
        ):
            result = memory_rebuild()

        assert "2 documents embedded" in result
        assert "1 failed" in result

    @patch("mnemon.server.Store")
    def test_empty_timeline(self, MockStore):
        MockStore.return_value.timeline.return_value = []

        mock_embed_fn = MagicMock()
        with patch.dict(
            "sys.modules",
            {"mnemon.embedder": MagicMock(embed_document=mock_embed_fn)},
        ):
            result = memory_rebuild()

        assert "0 documents embedded" in result
        assert "0 failed" in result


# ── profile_get ──────────────────────────────────────────────────────────────


class TestProfileGet:
    @patch("mnemon.server.Store")
    def test_no_data(self, MockStore):
        MockStore.return_value.timeline.return_value = []

        result = profile_get()
        assert "No profile data yet" in result

    @patch("mnemon.server.Store")
    def test_only_preferences(self, MockStore):
        pref = _make_document(title="Dark Mode", content="User prefers dark mode in all editors.")
        mock_store = MockStore.return_value
        mock_store.timeline.side_effect = lambda limit, ct: (
            [pref] if ct == "preference" else []
        )

        result = profile_get()
        assert "## Preferences" in result
        assert "**Dark Mode**" in result
        assert "## Key Decisions" not in result

    @patch("mnemon.server.Store")
    def test_only_decisions(self, MockStore):
        dec = _make_document(title="Use Python", content="Decided to use Python over TypeScript.")
        mock_store = MockStore.return_value
        mock_store.timeline.side_effect = lambda limit, ct: (
            [dec] if ct == "decision" else []
        )

        result = profile_get()
        assert "## Key Decisions" in result
        assert "**Use Python**" in result
        assert "## Preferences" not in result

    @patch("mnemon.server.Store")
    def test_both_preferences_and_decisions(self, MockStore):
        pref = _make_document(title="Vim Keys", content="Prefers vim keybindings.")
        dec = _make_document(title="Chose SQLite", content="SQLite over Postgres for simplicity.")
        mock_store = MockStore.return_value
        mock_store.timeline.side_effect = lambda limit, ct: (
            [pref] if ct == "preference" else [dec] if ct == "decision" else []
        )

        result = profile_get()
        assert "## Preferences" in result
        assert "## Key Decisions" in result
        assert "**Vim Keys**" in result
        assert "**Chose SQLite**" in result

    @patch("mnemon.server.Store")
    def test_long_content_truncated(self, MockStore):
        pref = _make_document(title="Long Pref", content="z" * 500)
        mock_store = MockStore.return_value
        mock_store.timeline.side_effect = lambda limit, ct: (
            [pref] if ct == "preference" else []
        )

        result = profile_get()
        # Content should be truncated to 200 chars
        assert "z" * 200 in result
        assert "z" * 201 not in result


# ── profile_update ───────────────────────────────────────────────────────────


class TestProfileUpdate:
    @patch("mnemon.server.Store")
    def test_success(self, MockStore):
        mock_store = MockStore.return_value
        mock_store.save.return_value = 15
        mock_store.get.return_value = _make_document(id=15, hash="h15")

        result = profile_update("Theme", "Prefers dark mode")
        assert result == 'Profile updated — saved preference #15: "Theme"'
        mock_store.save.assert_called_once_with(
            title="Theme",
            content="Prefers dark mode",
            content_type="preference",
            source_client="mcp-profile",
        )

    @patch("mnemon.server.Store")
    def test_embedding_failure_still_succeeds(self, MockStore):
        mock_store = MockStore.return_value
        mock_store.save.return_value = 20
        mock_store.get.return_value = _make_document(id=20, hash="h20")

        # Embedding import will fail — should still return success
        result = profile_update("Editor", "VS Code")
        assert 'Profile updated — saved preference #20: "Editor"' == result


# ── memory_check_contradictions ──────────────────────────────────────────────


class TestMemoryCheckContradictions:
    @patch("mnemon.server.Store")
    def test_not_found(self, MockStore):
        MockStore.return_value.get.return_value = None
        result = memory_check_contradictions(999)
        assert result == "Memory #999 not found."

    @patch("mnemon.server.check_contradictions", create=True)
    @patch("mnemon.server.Store")
    def test_no_contradictions(self, MockStore, mock_check):
        doc = _make_document(id=5, title="Some Fact", content="The sky is blue.")
        MockStore.return_value.get.return_value = doc

        mock_check_fn = MagicMock(return_value={"relationships": [], "decayed": 0})
        with patch.dict(
            "sys.modules",
            {"mnemon.contradiction": MagicMock(check_contradictions=mock_check_fn)},
        ):
            result = memory_check_contradictions(5)

        assert result == "No contradictions found for memory #5."

    @patch("mnemon.server.Store")
    def test_with_contradictions(self, MockStore):
        doc = _make_document(id=5, title="New Fact", content="Python is best.")
        MockStore.return_value.get.return_value = doc

        contradiction_result = {
            "relationships": [
                {"doc_id": 2, "title": "Old Fact", "relationship": "contradiction"},
                {"doc_id": 3, "title": "Similar Fact", "relationship": "update"},
            ],
            "decayed": 1,
        }
        mock_check_fn = MagicMock(return_value=contradiction_result)
        with patch.dict(
            "sys.modules",
            {"mnemon.contradiction": MagicMock(check_contradictions=mock_check_fn)},
        ):
            result = memory_check_contradictions(5)

        assert 'Contradiction check for #5 "New Fact"' in result
        assert '#2 "Old Fact" \u2192 **contradiction**' in result
        assert '#3 "Similar Fact" \u2192 **update**' in result
        assert "1 memories had their confidence decayed." in result

    @patch("mnemon.server.Store")
    def test_zero_decayed(self, MockStore):
        doc = _make_document(id=1, title="Fact", content="Content")
        MockStore.return_value.get.return_value = doc

        contradiction_result = {
            "relationships": [
                {"doc_id": 9, "title": "Same Thing", "relationship": "same"},
            ],
            "decayed": 0,
        }
        mock_check_fn = MagicMock(return_value=contradiction_result)
        with patch.dict(
            "sys.modules",
            {"mnemon.contradiction": MagicMock(check_contradictions=mock_check_fn)},
        ):
            result = memory_check_contradictions(1)

        assert "0 memories had their confidence decayed." in result
