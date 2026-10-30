"""Streamlit entry point."""

import streamlit as st

from yunex_demo.config import settings

st.set_page_config(page_title="Yunex Demo", page_icon=":material/network_node:", layout="wide")

st.title("Reliable LLM pipelines, running locally")
st.caption("Extract structured data, classify it, and act on it -- entirely on your own machine.")

st.markdown(
    """
This demo builds up in five steps, each a page in the sidebar:

1. **Bare chatbot** -- the model with only a system prompt: no tools, no
   schema, no grounding. The baseline everything after it improves on.
2. **One-off** -- a single schema-constrained call. Invalid output is made
   impossible rather than merely unlikely.
3. **Workflow** -- a deterministic pipeline that extracts, classifies, then
   acts, validating every step and abstaining instead of guessing.
4. **Embedding search** -- embed a question and pull the top-k most similar
   chunks from the vector store, to see what retrieval actually returns.
5. **RAG agent** -- retrieval over a local vector store, with a tool-using
   agent that cites what it used.

The whole stack is local: **Ollama** for inference, **ChromaDB** for vectors,
**EmbeddingGemma** for embeddings. Nothing leaves the machine.
"""
)

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Start here")
    st.markdown(
        "Open **Stack Health** in the sidebar to confirm every service is up "
        "and the embed/store/query chain works before you present."
    )
    st.page_link(
        "pages/0_Stack_Health.py", label="Stack Health", icon=":material/health_and_safety:"
    )

with right:
    st.subheader("Active configuration")
    st.markdown(
        f"""
| | |
|---|---|
| Chat model | `{settings.model_main}` |
| Embedding model | `{settings.model_embed}` |
| Ollama | `{settings.ollama_host}` |
| Chroma | `{settings.chroma_url}` |
| Context window | `{settings.ollama_num_ctx}` |
"""
    )
