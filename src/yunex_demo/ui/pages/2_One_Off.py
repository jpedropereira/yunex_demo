"""Demo beat 1: a single schema-constrained extraction call."""

import streamlit as st

st.set_page_config(page_title="One-off", page_icon=":material/bolt:", layout="wide")

st.title("One-off extraction")
st.caption("One document in, validated structured data out.")

st.markdown(
    """
This page will show the smallest reliable unit: a **single call constrained by a
JSON schema**, so malformed output is impossible rather than merely unlikely --
plus Pydantic validation, a bounded repair loop, and a grounding check that
verifies quoted spans actually appear in the source.
"""
)

st.info(
    "Not implemented yet -- infrastructure only. Logic is planned separately.",
    icon=":material/construction:",
)
