"""Unit tests for the dashboard chart builders (pure plotly, no Streamlit).

Skips cleanly where plotly isn't installed (the `[ui]` extra).
"""

from __future__ import annotations

import pytest

pytest.importorskip("plotly", reason="charts need plotly ([ui] extra)")

import numpy as np  # noqa: E402

from mnemon.dashboard.charts import make_graph_scatter  # noqa: E402


def _fixture(n: int = 6):
    vec_ids = [f"doc_{i}" for i in range(n)]
    doc_map = {
        vid: {
            "id": i,
            "title": f"Doc {i}",
            "content_type": "note",
            "confidence": 0.8,
            "created_at": "2026-05-01T00:00:00",
        }
        for i, vid in enumerate(vec_ids)
    }
    coords = np.array([[float(i), float(i)] for i in range(n)])
    return coords, vec_ids, doc_map


@pytest.mark.parametrize("projection", ["PCA", "UMAP"])
def test_title_reflects_projection(projection):
    coords, vec_ids, doc_map = _fixture()
    fig = make_graph_scatter(coords, vec_ids, doc_map, projection=projection)
    # The reducer must be named honestly — this is the rc18 regression
    # where the title said UMAP while remote mode actually ran PCA.
    assert projection in fig.layout.title.text


def test_legend_sits_below_plot_not_over_title():
    # The legend collided with the title when floated top-right; it must
    # now anchor below the plot (negative y).
    coords, vec_ids, doc_map = _fixture()
    fig = make_graph_scatter(coords, vec_ids, doc_map, projection="PCA")
    assert fig.layout.legend.y is not None and fig.layout.legend.y < 0


def test_default_projection_is_pca():
    # Remote (PCA) is the default path, so the unlabeled call should not
    # silently claim UMAP.
    coords, vec_ids, doc_map = _fixture()
    fig = make_graph_scatter(coords, vec_ids, doc_map)
    assert "PCA" in fig.layout.title.text
