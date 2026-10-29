"""Demo beat 2: a deterministic extract -> classify -> act pipeline."""

import streamlit as st

st.set_page_config(page_title="Workflow", page_icon=":material/account_tree:", layout="wide")

st.title("Workflow")
st.caption("Deterministic steps, validated individually, with a visible trace.")

st.markdown(
    """
This page will chain the one-off call into a **fixed pipeline**: extract, then
classify against a closed taxonomy, then route or act. Each step is small,
separately validated, and recorded in a trace -- and the pipeline abstains into
a review queue rather than guessing when confidence is low.
"""
)

st.info(
    "Not implemented yet -- infrastructure only. Logic is planned separately.",
    icon=":material/construction:",
)
