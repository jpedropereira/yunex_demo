"""The two services behind semantic search: the embedding model and the vector store."""

import contextlib

import chromadb
from chromadb.errors import NotFoundError
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


def _client() -> chromadb.ClientAPI:
    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


def _open_collection(name: str) -> chromadb.Collection:
    return _client().get_or_create_collection(
        name=name,
        # We supply vectors ourselves, and must not let Chroma fall back to
        # downloading its default ONNX embedder.
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def get_collection() -> chromadb.Collection:
    """The collection of SEC filings. Raises ValueError if Chroma is unreachable."""
    return _open_collection(settings.sec_collection)


def _delete_if_present(name: str) -> None:
    with contextlib.suppress(NotFoundError):
        _client().delete_collection(name)


def empty_rebuild_collection() -> chromadb.Collection:
    """A collection for ``make reindex`` to fill, replacing any that a failed run left behind."""
    name = f"{settings.sec_collection}_rebuild"
    _delete_if_present(name)
    return _open_collection(name)


def replace_index(rebuilt: chromadb.Collection) -> None:
    """Make a finished rebuild the live index. Renaming keeps its vectors, so nothing re-embeds."""
    _delete_if_present(settings.sec_collection)
    rebuilt.modify(name=settings.sec_collection)
