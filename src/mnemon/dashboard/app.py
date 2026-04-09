"""mnemon Dashboard — Home / Vault Health."""

import streamlit as st

st.set_page_config(
    page_title="mnemon — Memory Vault",
    layout="wide",
    initial_sidebar_state="expanded",
)

from .loaders import load_status, load_timeline, load_sweep
from .charts import make_type_distribution_chart, make_accumulation_chart


def _render_metrics(status: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Memories", status["total_documents"])
    c2.metric("Vectors", status["total_vectors"])
    c3.metric("Pinned", status["pinned"])
    c4.metric("Invalidated", status["invalidated"])


def _render_sweep_candidates(sweep: dict) -> None:
    candidates = sweep.get("candidates", [])
    if not candidates:
        st.info("No stale memories found.")
        return
    st.dataframe(
        [{"Title": c["title"], "Type": c["content_type"], "Age (days)": c["age_days"]} for c in candidates],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.title("mnemon — Memory Vault")
    st.caption("Long-term memory for AI agents")

    status = load_status()
    if not status or status["total_documents"] == 0:
        st.warning("Vault is empty. Start saving memories with `mnemon save` or via MCP tools.")
        st.stop()

    st.subheader("Vault Health")
    _render_metrics(status)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig = make_type_distribution_chart(status.get("by_type", []))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        timeline = load_timeline(limit=500)
        fig = make_accumulation_chart(timeline)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Stale Memory Candidates")
    sweep = load_sweep()
    _render_sweep_candidates(sweep)


main()
