"""Plotly chart builders for the dashboard."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go

CONTENT_TYPE_COLORS = {
    "decision": "#636EFA",
    "preference": "#EF553B",
    "antipattern": "#FFA15A",
    "observation": "#00CC96",
    "research": "#AB63FA",
    "project": "#19D3F3",
    "handoff": "#FF6692",
    "note": "#B6E880",
}

_LAYOUT_DEFAULTS = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=40, b=30, l=40, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def make_type_distribution_chart(by_type: list[dict]) -> go.Figure:
    """Donut chart of memory count by content_type."""
    if not by_type:
        fig = go.Figure()
        fig.update_layout(title="No data", **_LAYOUT_DEFAULTS)
        return fig

    labels = [t["content_type"] for t in by_type]
    values = [t["count"] for t in by_type]
    colors = [CONTENT_TYPE_COLORS.get(label, "#999") for label in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo="label+value",
        hovertemplate="%{label}: %{value} memories<extra></extra>",
    )])
    fig.update_layout(title="Memory Types", showlegend=False, height=350, **_LAYOUT_DEFAULTS)
    return fig


def make_accumulation_chart(timeline_docs: list[dict]) -> go.Figure:
    """Bar chart: memories saved per day over last 30 days."""
    if not timeline_docs:
        fig = go.Figure()
        fig.update_layout(title="No data", **_LAYOUT_DEFAULTS)
        return fig

    cutoff = datetime.now() - timedelta(days=30)
    dates = []
    for d in timeline_docs:
        created = datetime.fromisoformat(d["created_at"])
        if created >= cutoff:
            dates.append(created.date())

    if not dates:
        fig = go.Figure()
        fig.update_layout(title="No memories in last 30 days", **_LAYOUT_DEFAULTS)
        return fig

    counts = Counter(dates)
    sorted_dates = sorted(counts.keys())
    x = [str(d) for d in sorted_dates]
    y = [counts[d] for d in sorted_dates]

    fig = go.Figure(data=[go.Bar(x=x, y=y, marker_color="#636EFA")])
    fig.update_layout(title="Memories Saved (Last 30 Days)", xaxis_title="Date", yaxis_title="Count", height=350, **_LAYOUT_DEFAULTS)
    return fig


def make_score_bars(composite: float, recency: float, confidence: float) -> go.Figure:
    """Horizontal progress bars for search result score breakdown."""
    fig = go.Figure()
    labels = ["Relevance", "Recency", "Confidence"]
    values = [composite, recency, confidence]
    colors = ["#636EFA", "#00CC96", "#FFA15A"]

    for label, val, color in zip(labels, values, colors):
        fig.add_trace(go.Bar(
            y=[label], x=[val], orientation="h",
            marker_color=color, name=label,
            text=[f"{val:.3f}"], textposition="auto",
            showlegend=False,
        ))

    fig.update_layout(
        height=120,
        xaxis=dict(range=[0, 1], showticklabels=False),
        yaxis=dict(autorange="reversed"),
        bargap=0.3,
        margin=dict(t=5, b=5, l=80, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_graph_scatter(
    coords_2d: np.ndarray,
    vec_ids: list[str],
    doc_map: dict[str, dict],
    visible_types: set[str] | None = None,
) -> go.Figure:
    """UMAP scatter plot — one trace per content_type."""
    fig = go.Figure()

    type_groups: dict[str, dict] = {}
    unmapped = {"x": [], "y": [], "text": [], "customdata": []}

    for i, vid in enumerate(vec_ids):
        x, y = float(coords_2d[i, 0]), float(coords_2d[i, 1])
        info = doc_map.get(vid)

        if info is None:
            unmapped["x"].append(x)
            unmapped["y"].append(y)
            unmapped["text"].append(f"Orphan vector: {vid[:16]}...")
            unmapped["customdata"].append({"vec_id": vid})
            continue

        ct = info["content_type"]
        if visible_types and ct not in visible_types:
            continue

        if ct not in type_groups:
            type_groups[ct] = {"x": [], "y": [], "text": [], "customdata": []}
        g = type_groups[ct]
        g["x"].append(x)
        g["y"].append(y)
        g["text"].append(
            f"<b>{info['title']}</b><br>"
            f"Type: {ct}<br>"
            f"Confidence: {info['confidence']:.0%}<br>"
            f"Created: {info['created_at']}"
        )
        g["customdata"].append({"doc_id": info["id"], "vec_id": vid})

    for ct, g in type_groups.items():
        fig.add_trace(go.Scatter(
            x=g["x"], y=g["y"],
            mode="markers",
            name=ct,
            marker=dict(color=CONTENT_TYPE_COLORS.get(ct, "#999"), size=8, opacity=0.7),
            hovertemplate="%{text}<extra></extra>",
            text=g["text"],
            customdata=g["customdata"],
        ))

    if unmapped["x"]:
        fig.add_trace(go.Scatter(
            x=unmapped["x"], y=unmapped["y"],
            mode="markers",
            name="unmapped",
            marker=dict(color="#666", size=5, opacity=0.3),
            hovertemplate="%{text}<extra></extra>",
            text=unmapped["text"],
            customdata=unmapped["customdata"],
        ))

    fig.update_layout(
        title="Memory Vector Space (UMAP 2D)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=650,
        **_LAYOUT_DEFAULTS,
    )
    return fig


def add_relation_edges(
    fig: go.Figure,
    coords_2d: np.ndarray,
    vec_ids: list[str],
    doc_map: dict[str, dict],
    relations: dict[int, list[dict]],
) -> go.Figure:
    """Overlay relation edges as semi-transparent lines."""
    doc_id_to_idx: dict[int, int] = {}
    for i, vid in enumerate(vec_ids):
        info = doc_map.get(vid)
        if info and info["id"] not in doc_id_to_idx:
            doc_id_to_idx[info["id"]] = i

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []

    for source_id, rels in relations.items():
        if source_id not in doc_id_to_idx:
            continue
        src_idx = doc_id_to_idx[source_id]
        for rel in rels:
            target_id = rel.get("id")
            if target_id not in doc_id_to_idx:
                continue
            tgt_idx = doc_id_to_idx[target_id]
            edge_x.extend([float(coords_2d[src_idx, 0]), float(coords_2d[tgt_idx, 0]), None])
            edge_y.extend([float(coords_2d[src_idx, 1]), float(coords_2d[tgt_idx, 1]), None])

    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            name="relations",
            line=dict(color="rgba(150,150,150,0.3)", width=1),
            hoverinfo="skip",
        ))
    return fig
