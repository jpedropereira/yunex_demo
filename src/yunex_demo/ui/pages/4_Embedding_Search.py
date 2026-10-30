"""Demo beat 4: embed a question and retrieve the top-k chunks from ChromaDB."""

import streamlit as st

st.set_page_config(
    page_title="Embedding search", page_icon=":material/scatter_plot:", layout="wide"
)

st.title("Embedding search")
st.caption("A question in, the most similar chunks from the vector store out.")

st.markdown(
    """
This page will show **retrieval on its own**, before an agent uses it: the
question is embedded with EmbeddingGemma into a vector, ChromaDB returns the
**top-k nearest chunks**, and each one is shown with its similarity score --
so you can see what the model *would* be given as context, and judge whether
it is relevant before any answer is generated.
"""
)

st.info(
    "Not implemented yet -- infrastructure only. Logic is planned separately.",
    icon=":material/construction:",
)
