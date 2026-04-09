"""Timeline — chronological memory feed with filters."""

import streamlit as st
from datetime import datetime, date

st.set_page_config(page_title="Timeline — mnemon", layout="wide")
st.title("Memory Timeline")

from mnemon.dashboard.loaders import load_timeline
from mnemon.dashboard.charts import CONTENT_TYPE_COLORS

CONTENT_TYPES = ["decision", "preference", "antipattern", "observation", "research", "project", "handoff", "note"]

types = st.sidebar.multiselect("Content types", CONTENT_TYPES, default=CONTENT_TYPES)
col1, col2 = st.sidebar.columns(2)
date_from = col1.date_input("From", value=date(2024, 1, 1))
date_to = col2.date_input("To", value=date.today())

timeline = load_timeline(limit=500)

filtered = [
    d for d in timeline
    if d["content_type"] in types
    and date_from <= datetime.fromisoformat(d["created_at"]).date() <= date_to
]

if not filtered:
    st.info("No memories match your filters.")
    st.stop()

st.caption(f"Showing {len(filtered)} memories")

for doc in filtered:
    color = CONTENT_TYPE_COLORS.get(doc["content_type"], "#999")
    pinned = " (pinned)" if doc.get("pinned") else ""
    header = f"**{doc['title']}** — :{color}[{doc['content_type']}] — confidence: {doc['confidence']:.0%}{pinned}"

    with st.expander(header):
        st.caption(f"Created: {doc['created_at']} | ID: {doc['id']} | Accessed: {doc.get('access_count', 0)}x")
        st.markdown(doc["content"])
