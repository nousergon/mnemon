"""Search — hybrid BM25 + vector search with score breakdown."""

import streamlit as st

st.set_page_config(page_title="Search — mnemon", layout="wide")
st.title("Search Memories")

from mnemon.dashboard.loaders import load_search, remote_guard
from mnemon.dashboard.charts import make_score_bars

CONTENT_TYPES = ["All", "decision", "preference", "antipattern", "observation", "research", "project", "handoff", "note"]

query = st.text_input("Search query", placeholder="e.g. deployment architecture")
content_type_filter = st.selectbox("Filter by type", CONTENT_TYPES)

if query:
    ct = None if content_type_filter == "All" else content_type_filter
    with remote_guard("search results"):
        results = load_search(query, limit=20, content_type=ct)

    if not results:
        st.info("No results found.")
    else:
        st.caption(f"{len(results)} results")
        for r in results:
            with st.expander(f"**{r['title']}** — `{r['content_type']}` — score: {r.get('composite_score', 0.0):.3f}"):
                # .get() defensively: an older remote (e.g. a Fly app on a
                # pre-recency_score mnemon) won't include every score field.
                fig = make_score_bars(
                    r.get("composite_score", 0.0),
                    r.get("recency_score", 0.0),
                    r.get("confidence", 0.0),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"score_{r['doc_id']}")
                st.markdown(r["content"])
                st.caption(f"ID: {r['doc_id']} | Created: {r['created_at']}")
