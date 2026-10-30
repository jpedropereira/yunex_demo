"""Demo beat 3: retrieval-augmented agent over the local vector store."""

import streamlit as st

st.set_page_config(page_title="RAG Agent", page_icon=":material/search:", layout="wide")

st.title("RAG agent")
st.caption("Retrieval over ChromaDB, with tool use and citations.")

st.markdown(
    """
This page will add **retrieval and agency**: a corpus embedded with
EmbeddingGemma into ChromaDB, and an agent that decides when to search it,
cites what it used, and has a capped iteration budget so it cannot loop
indefinitely.
"""
)

st.info(
    "Not implemented yet -- infrastructure only. Logic is planned separately.",
    icon=":material/construction:",
)
