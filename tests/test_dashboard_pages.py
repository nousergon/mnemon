"""Headless render smoke tests for the Streamlit dashboard pages.

Uses Streamlit's ``AppTest`` (in-process, no browser, no server) to render
each dashboard page against **remote-shaped** mocked loader output, then
asserts the page rendered without an exception.

Why this matters: the dashboard has two data paths — local
(``dataclasses.asdict`` → full fields) and remote (MCP-tool JSON → only
the fields each tool serializes). A page that reads a field the *remote*
JSON omits crashes only in remote mode. That's exactly the
``KeyError: 'recency_score'`` bug (rc14): ``memory_search`` hand-built its
dict and dropped a field the Search page read. These tests render every
page with remote-shaped data so that whole class is caught in CI.

Skips cleanly where Streamlit isn't installed (e.g. the [server]-only CI
job) via ``importorskip`` — the dashboard is the ``[ui]`` extra.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit", reason="dashboard tests need the [ui] extra (streamlit)")
pytest.importorskip("plotly", reason="dashboard charts need plotly ([ui] extra)")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASH = Path(__file__).resolve().parents[1] / "src" / "mnemon" / "dashboard"
PAGES = DASH / "pages"
RUN_TIMEOUT = 30  # AppTest default is 3s; UMAP/plotly builds can exceed it


# ── remote-shaped fixtures (mirror what the MCP tools serialize) ─────────────

def _timeline_doc(**ov):
    d = {
        "id": 1,
        "title": "A memory",
        "content": "Some content body.",
        "content_type": "note",
        "confidence": 0.8,
        "created_at": "2026-05-01T00:00:00",
        "pinned": 0,
        "access_count": 3,
    }
    d.update(ov)
    return d


def _search_result(**ov):
    r = {
        "doc_id": 1,
        "title": "A result",
        "content": "Result content.",
        "content_type": "decision",
        "confidence": 0.9,
        "composite_score": 0.7,
        "recency_score": 0.5,
        "vector_similarity": 0.88,
        "created_at": "2026-05-01T00:00:00",
    }
    r.update(ov)
    return r


def _status():
    return {
        "total_documents": 12,
        "total_vectors": 14,
        "pinned": 2,
        "invalidated": 1,
        "by_type": [
            {"content_type": "note", "count": 7},
            {"content_type": "decision", "count": 5},
        ],
    }


def _sweep():
    return {"candidates": [{"title": "Stale one", "content_type": "note", "age_days": 95}]}


def _assert_clean(at: AppTest):
    assert not at.exception, f"page raised: {at.exception}"


def _assert_degraded(at: AppTest):
    """A loader failure must degrade to a clean on-page error (the
    remote_guard message starts 'Couldn't load …') + no raw exception."""
    assert not at.exception, f"page raised instead of degrading: {at.exception}"
    assert any("Couldn't load" in e.value for e in at.error), \
        "expected a clean 'Couldn't load …' error from remote_guard"


# ── Home (app.py) ────────────────────────────────────────────────────────────

def test_home_renders():
    with patch("mnemon.dashboard.loaders.load_status", return_value=_status()), \
         patch("mnemon.dashboard.loaders.load_timeline", return_value=[_timeline_doc()]), \
         patch("mnemon.dashboard.loaders.load_sweep", return_value=_sweep()), \
         patch("mnemon.dashboard.loaders._use_remote", return_value=True):
        at = AppTest.from_file(str(DASH / "app.py"), default_timeout=RUN_TIMEOUT).run()
    _assert_clean(at)


def test_home_empty_vault_is_graceful():
    empty = {**_status(), "total_documents": 0}
    with patch("mnemon.dashboard.loaders.load_status", return_value=empty), \
         patch("mnemon.dashboard.loaders._use_remote", return_value=False):
        at = AppTest.from_file(str(DASH / "app.py"), default_timeout=RUN_TIMEOUT).run()
    _assert_clean(at)


# ── Search (1_Search.py) ─────────────────────────────────────────────────────

def test_search_renders_results():
    with patch("mnemon.dashboard.loaders.load_search", return_value=[_search_result()]):
        at = AppTest.from_file(str(PAGES / "1_Search.py"), default_timeout=RUN_TIMEOUT).run()
        at.text_input[0].set_value("anything").run()
    _assert_clean(at)


def test_search_tolerates_older_remote_missing_recency_score():
    # The exact rc14 regression: an older remote omits recency_score. The
    # page must NOT crash (it reads score fields with .get()).
    legacy = _search_result()
    del legacy["recency_score"]
    with patch("mnemon.dashboard.loaders.load_search", return_value=[legacy]):
        at = AppTest.from_file(str(PAGES / "1_Search.py"), default_timeout=RUN_TIMEOUT).run()
        at.text_input[0].set_value("anything").run()
    _assert_clean(at)


# ── Timeline (2_Timeline.py) ─────────────────────────────────────────────────

def test_timeline_renders():
    with patch("mnemon.dashboard.loaders.load_timeline", return_value=[_timeline_doc()]):
        at = AppTest.from_file(str(PAGES / "2_Timeline.py"), default_timeout=RUN_TIMEOUT).run()
    _assert_clean(at)


# ── Profile (4_Profile.py) ───────────────────────────────────────────────────

def test_profile_renders():
    with patch(
        "mnemon.dashboard.loaders.load_timeline",
        return_value=[_timeline_doc(content_type="preference")],
    ):
        at = AppTest.from_file(str(PAGES / "4_Profile.py"), default_timeout=RUN_TIMEOUT).run()
    _assert_clean(at)


# ── Graph (3_Graph.py) ───────────────────────────────────────────────────────

def _graph_doc_map(vec_ids):
    return {
        vid: {
            "id": i,
            "title": f"Doc {i}",
            "content_type": "note",
            "confidence": 0.8,
            "created_at": "2026-05-01T00:00:00",
        }
        for i, vid in enumerate(vec_ids)
    }


def test_graph_empty_is_graceful():
    with patch("mnemon.dashboard.loaders._use_remote", return_value=False), \
         patch("mnemon.dashboard.loaders.load_status", return_value={"total_documents": 0}):
        at = AppTest.from_file(str(PAGES / "3_Graph.py"), default_timeout=RUN_TIMEOUT).run()
    _assert_clean(at)


def test_graph_renders_with_points_local_umap():
    import numpy as np

    vec_ids = [f"v{i}" for i in range(6)]
    doc_map = _graph_doc_map(vec_ids)
    # load_umap_coords returns an (N, 2) numpy array (charts index it as [i, 0]).
    coords = np.array([[float(i), float(i)] for i in range(6)])
    with patch("mnemon.dashboard.loaders._use_remote", return_value=False), \
         patch("mnemon.dashboard.loaders.load_status", return_value={"total_documents": 6}), \
         patch(
             "mnemon.dashboard.loaders.load_vectors_collapsed",
             return_value=(vec_ids, np.zeros((6, 4)), doc_map),
         ), \
         patch("mnemon.dashboard.loaders.load_umap_coords", return_value=coords), \
         patch("mnemon.dashboard.loaders.load_related", return_value=[]):
        at = AppTest.from_file(str(PAGES / "3_Graph.py"), default_timeout=RUN_TIMEOUT).run()
    _assert_clean(at)


def test_graph_renders_with_points_remote_pca():
    # Remote path: server-side PCA coords come back via load_coords_remote
    # as (vec_ids, (N,2) coords, doc_map). The page must render the same.
    import numpy as np

    vec_ids = [f"doc_{i}" for i in range(6)]
    doc_map = _graph_doc_map(vec_ids)
    coords = np.array([[float(i), -float(i)] for i in range(6)], dtype=np.float32)
    with patch("mnemon.dashboard.loaders._use_remote", return_value=True), \
         patch("mnemon.dashboard.loaders.load_status", return_value={"total_documents": 6}), \
         patch(
             "mnemon.dashboard.loaders.load_coords_remote",
             return_value=(vec_ids, coords, doc_map),
         ), \
         patch("mnemon.dashboard.loaders.load_related", return_value=[]):
        at = AppTest.from_file(str(PAGES / "3_Graph.py"), default_timeout=RUN_TIMEOUT).run()
    _assert_clean(at)


# ── loader-FAILURE paths (remote_guard) — every page must degrade, not crash ──
# A remote-call failure (timeout / transport ExceptionGroup on a slow/cold/
# unreachable remote) must surface as a clean on-page error, never a traceback.
# These are the paths the success-mocked render tests above don't exercise.

def test_graph_degrades_when_local_projection_fails():
    with patch("mnemon.dashboard.loaders._use_remote", return_value=False), \
         patch("mnemon.dashboard.loaders.load_status", return_value={"total_documents": 6}), \
         patch(
             "mnemon.dashboard.loaders.load_vectors_collapsed",
             side_effect=RuntimeError("boom: vector export failed"),
         ):
        at = AppTest.from_file(str(PAGES / "3_Graph.py"), default_timeout=RUN_TIMEOUT).run()
    _assert_degraded(at)


def test_graph_degrades_when_remote_coords_fail():
    # The exact failure class Brian hit: the heavy remote projection call
    # times out / transport-errors on a cold or large remote.
    with patch("mnemon.dashboard.loaders._use_remote", return_value=True), \
         patch("mnemon.dashboard.loaders.load_status", return_value={"total_documents": 3057}), \
         patch(
             "mnemon.dashboard.loaders.load_coords_remote",
             side_effect=RuntimeError("remote coords export failed"),
         ):
        at = AppTest.from_file(str(PAGES / "3_Graph.py"), default_timeout=RUN_TIMEOUT).run()
    _assert_degraded(at)


def test_search_degrades_on_loader_failure():
    with patch("mnemon.dashboard.loaders.load_search", side_effect=RuntimeError("remote down")):
        at = AppTest.from_file(str(PAGES / "1_Search.py"), default_timeout=RUN_TIMEOUT).run()
        at.text_input[0].set_value("anything").run()
    _assert_degraded(at)


def test_timeline_degrades_on_loader_failure():
    with patch("mnemon.dashboard.loaders.load_timeline", side_effect=RuntimeError("remote down")):
        at = AppTest.from_file(str(PAGES / "2_Timeline.py"), default_timeout=RUN_TIMEOUT).run()
    _assert_degraded(at)


def test_profile_degrades_on_loader_failure():
    with patch("mnemon.dashboard.loaders.load_timeline", side_effect=RuntimeError("remote down")):
        at = AppTest.from_file(str(PAGES / "4_Profile.py"), default_timeout=RUN_TIMEOUT).run()
    _assert_degraded(at)


def test_home_degrades_on_loader_failure():
    with patch("mnemon.dashboard.loaders.load_status", side_effect=RuntimeError("remote down")):
        at = AppTest.from_file(str(DASH / "app.py"), default_timeout=RUN_TIMEOUT).run()
    _assert_degraded(at)
