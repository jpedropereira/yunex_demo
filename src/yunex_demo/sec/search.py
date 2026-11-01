"""Find the pages closest in meaning to a question. No LLM: results come only from the database.

The agent page can reuse ``search`` as its ``search_documents`` tool.
"""

from dataclasses import dataclass

import chromadb

from yunex_demo.sec import store


@dataclass(frozen=True)
class SearchHit:
    similarity: float
    text: str
    company: str
    form: str
    fiscal_year: str
    page: int
    part: int
    source_url: str


@dataclass(frozen=True)
class IndexedDocument:
    doc_id: str
    company: str
    form: str
    fiscal_year: str
    pages: int
    source_url: str

    @property
    def label(self) -> str:
        return f"{self.company} · {self.form} · FY{self.fiscal_year}"


def search(
    question: str,
    k: int,
    collection: chromadb.Collection,
    doc_id: str | None = None,
) -> list[SearchHit]:
    """Return the ``k`` most similar chunks, best first, optionally from one document."""
    query_vector = store.embed_texts([store.query_prompt(question)])[0]

    # Chroma applies the filter before ranking: k results from this document, rather
    # than the overall top k with the other documents' results thrown away.
    where = {"doc_id": doc_id} if doc_id else None
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma answers per query vector, and we sent one.
    hits = []
    for text, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0], strict=True
    ):
        hits.append(
            SearchHit(
                similarity=round(1 - distance, 2),
                text=text,
                company=metadata["company"],
                form=metadata["form"],
                fiscal_year=metadata["fiscal_year"],
                page=metadata["page"],
                part=metadata["part"],
                source_url=metadata["source_url"],
            )
        )
    return hits


def list_documents(collection: chromadb.Collection) -> list[IndexedDocument]:
    """Every indexed filing. Chroma has no GROUP BY, so group the chunks' metadata here."""
    metadatas = collection.get(include=["metadatas"])["metadatas"]

    pages_by_doc: dict[str, set[int]] = {}
    first_chunk_by_doc = {}
    for metadata in metadatas:
        doc_id = metadata["doc_id"]
        pages_by_doc.setdefault(doc_id, set()).add(metadata["page"])
        first_chunk_by_doc.setdefault(doc_id, metadata)

    documents = [
        IndexedDocument(
            doc_id=doc_id,
            company=metadata["company"],
            form=metadata["form"],
            fiscal_year=metadata["fiscal_year"],
            pages=len(pages_by_doc[doc_id]),
            source_url=metadata["source_url"],
        )
        for doc_id, metadata in first_chunk_by_doc.items()
    ]
    return sorted(documents, key=lambda document: document.label)
