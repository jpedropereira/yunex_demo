"""The two services behind semantic search: the embedding model and the vector store."""

import chromadb
from ollama import Client

from yunex_demo.config import settings


# EmbeddingGemma embeds documents and queries differently, and its task prefixes say
# which is which. Both ingest and search must use them, or precision drops.
def document_prompt(title: str, text: str) -> str:
    return f"title: {title} | text: {text}"


def query_prompt(question: str) -> str:
    return f"task: search result | query: {question}"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of already-prefixed texts, one vector per text."""
    client = Client(host=settings.ollama_host, timeout=600)
    response = client.embed(
        model=settings.model_embed,
        input=texts,
        # Over-long input would otherwise lose its end without any warning. That text
        # could never be found, so make Ollama return an error instead.
        truncate=False,
        keep_alive=settings.ollama_keep_alive,
    )
    return response.embeddings


def get_collection() -> chromadb.Collection:
    """The collection of SEC filings. Raises ValueError if Chroma is unreachable."""
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return client.get_or_create_collection(
        name=settings.sec_collection,
        # We supply vectors ourselves, and must not let Chroma fall back to
        # downloading its default ONNX embedder.
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
