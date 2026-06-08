"""Memory Graph — UMAP 2D projection of the vector space."""

import streamlit as st

st.set_page_config(page_title="Memory Graph — mnemon", layout="wide")
st.title("Memory Graph")

from mnemon.dashboard.loaders import load_graph_projection, load_relations_bulk, load_related, load_status, remote_guard, _use_remote
from mnemon.dashboard.charts import make_graph_scatter, add_relation_edges

CONTENT_TYPES = ["decision", "preference", "antipattern", "observation", "research", "project", "handoff", "note"]

remote = _use_remote()

# Status first (cheap) — bounds the empty-vault case and the local UMAP
# n_neighbors slider without paying for the projection up front.
with remote_guard("vault status"):
    doc_count = load_status().get("total_documents", 0)
if doc_count == 0:
    st.warning("No memories saved yet. Save a memory to get started.")
    st.stop()

# Sidebar controls. The neighbor knob is UMAP-only (local); the remote
# path is server-side PCA, which ships just the 2-D coordinates so the
# Graph scales to large vaults instead of timing out on a full export.
if remote:
    st.sidebar.caption("Projection: PCA — computed server-side, scales to large vaults.")
    n_neighbors = 15  # unused on the remote (PCA) path
else:
    max_neighbors = min(50, max(2, doc_count - 1))
    if max_neighbors > 5:
        n_neighbors = st.sidebar.slider("UMAP n_neighbors", min_value=5, max_value=max_neighbors, value=min(15, max_neighbors), step=5, help="Higher = more global structure, lower = more local clusters")
    else:
        n_neighbors = max_neighbors
        st.sidebar.caption(f"UMAP n_neighbors: {n_neighbors} (auto — small vault)")
visible_types = st.sidebar.multiselect("Visible types", CONTENT_TYPES, default=CONTENT_TYPES)
show_edges = st.sidebar.checkbox("Show relation edges", value=True)

# One point per document. Multi-chunk docs are mean-pooled (so a long
# memory isn't several near-identical points). Remote reduces server-side
# (tiny payload); local runs UMAP client-side — both can be slow/cold, so
# guard the call.
with remote_guard("the memory graph projection"):
    with st.spinner("Computing projection..."):
        proj = load_graph_projection(n_neighbors=n_neighbors)

if proj is None:
    # doc_count > 0 but no embeddings — a silent-failure signal; tell the
    # user how to fix it rather than showing a blank graph.
    st.warning(
        f"{doc_count} memor{'y' if doc_count == 1 else 'ies'} saved but no embeddings found — "
        "the embedding step did not run or failed silently. "
        "Run `mnemon doctor` to diagnose, then `mnemon rebuild` to re-embed."
    )
    st.stop()

vec_ids, coords_2d, doc_map = proj

# Post-collapse each point = one memory, so count in memory-units here —
# otherwise the Graph page's "1 vector" contradicts `mnemon status`'s
# "2 vectors" (raw chunk count: full doc + section per save).
if len(vec_ids) < 5:
    st.warning(
        f"Only {len(vec_ids)} memor{'y' if len(vec_ids) == 1 else 'ies'} — "
        "need at least 5 memories for a meaningful projection. "
        "Add more and come back."
    )
    st.stop()

# Build scatter plot
fig = make_graph_scatter(
    coords_2d, vec_ids, doc_map,
    visible_types=set(visible_types),
    projection="PCA" if remote else "UMAP",
)

# Relation edges — one bulk fetch (was one call per document, which
# timed out on large remotes). add_relation_edges filters to visible nodes.
if show_edges:
    with remote_guard("relation edges"):
        relations = load_relations_bulk()
    if relations:
        fig = add_relation_edges(fig, coords_2d, vec_ids, doc_map, relations)

# Render
selected = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="graph_scatter")

# Click detail
if selected and selected.selection and selected.selection.points:
    point = selected.selection.points[0]
    custom = point.get("customdata")
    if custom and isinstance(custom, dict) and "doc_id" in custom:
        from mnemon.dashboard.loaders import load_document
        with st.sidebar:
            st.divider()
            st.subheader("Memory Detail")
            doc = load_document(custom["doc_id"])
            if doc:
                st.markdown(f"**{doc['title']}**")
                st.caption(f"{doc['content_type']} | confidence: {doc['confidence']:.0%} | {doc['created_at']}")
                st.markdown(doc["content"])
                related = load_related(custom["doc_id"])
                if related:
                    st.subheader("Related")
                    for r in related:
                        st.markdown(f"- **{r['title']}** ({r.get('relation_type', '')}, weight: {r.get('weight', 0):.2f})")
