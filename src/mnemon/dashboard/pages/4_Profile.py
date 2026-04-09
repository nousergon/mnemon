"""Profile — preferences and decisions."""

import streamlit as st

st.set_page_config(page_title="Profile — mnemon", layout="wide")
st.title("Your Profile")

from mnemon.dashboard.loaders import load_timeline

col1, col2 = st.columns(2)

with col1:
    st.subheader("Preferences")
    prefs = load_timeline(limit=50, content_type="preference")
    if not prefs:
        st.info("No preferences stored yet.")
    for p in prefs:
        with st.expander(f"**{p['title']}** — confidence: {p['confidence']:.0%}"):
            st.markdown(p["content"])
            st.caption(f"Created: {p['created_at']} | ID: {p['id']}")

with col2:
    st.subheader("Decisions")
    decisions = load_timeline(limit=50, content_type="decision")
    if not decisions:
        st.info("No decisions stored yet.")
    for d in decisions:
        with st.expander(f"**{d['title']}** — confidence: {d['confidence']:.0%}"):
            st.markdown(d["content"])
            st.caption(f"Created: {d['created_at']} | ID: {d['id']}")
