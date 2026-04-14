"""Memory Graph — UMAP 2D projection of the vector space."""

import streamlit as st

st.set_page_config(page_title="Memory Graph — mnemon", layout="wide")
st.title("Memory Graph")

from mnemon.dashboard.loaders import load_vectors, load_umap_coords, load_related
from mnemon.dashboard.charts import make_graph_scatter, add_relation_edges

CONTENT_TYPES = ["decision", "preference", "antipattern", "observation", "research", "project", "handoff", "note"]

# Load vectors — post-0.5.0 loaders.load_vectors returns a 3-tuple with
# the vec_id → doc map baked in (remote: from memory_export_vectors,
# local: joined from SQLite).
vec_data = load_vectors()
if vec_data is None:
    st.warning("No vectors found. Save some memories with embeddings first.")
    st.stop()

vec_ids, vectors, doc_map = vec_data

if len(vec_ids) < 5:
    st.warning(f"Only {len(vec_ids)} vectors — need at least 5 for UMAP projection.")
    st.stop()

# Sidebar controls
max_neighbors = min(50, len(vec_ids) - 1)
if max_neighbors > 5:
    n_neighbors = st.sidebar.slider("UMAP n_neighbors", min_value=5, max_value=max_neighbors, value=min(15, max_neighbors), step=5, help="Higher = more global structure, lower = more local clusters")
else:
    n_neighbors = max_neighbors
    st.sidebar.caption(f"UMAP n_neighbors: {n_neighbors} (auto — small vault)")
visible_types = st.sidebar.multiselect("Visible types", CONTENT_TYPES, default=CONTENT_TYPES)
show_edges = st.sidebar.checkbox("Show relation edges", value=True)

# UMAP reduction
with st.spinner("Computing UMAP projection..."):
    coords_2d = load_umap_coords(vectors, n_neighbors=n_neighbors)

# doc_map is already populated by load_vectors() — no extra query needed.

# Build scatter plot
fig = make_graph_scatter(coords_2d, vec_ids, doc_map, visible_types=set(visible_types))

# Relation edges
if show_edges:
    all_doc_ids = {info["id"] for info in doc_map.values()}
    relations: dict[int, list[dict]] = {}
    for doc_id in all_doc_ids:
        rels = load_related(doc_id, limit=5)
        if rels:
            relations[doc_id] = rels
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
