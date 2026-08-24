"""Top-k semantic retrieval over the chunk store, with optional metadata filters.

ChromaDB returns cosine distance (0 = identical direction, 2 = opposite); `RetrievedChunk.score`
is `1 - distance` instead, so "higher is more relevant" holds everywhere downstream (the eval
harness, the sources panel, the retrieval benchmark) without every caller re-deriving the sign.
"""

from dataclasses import dataclass

from src.store import embed_texts, get_collection


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    section: str
    score: float


def retrieve(query: str, k: int = 5, doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
    collection = get_collection()
    query_embedding = embed_texts([query])[0]

    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"][0]:
        return []

    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=metadata["doc_id"],
            text=document,
            page_start=metadata["page_start"],
            page_end=metadata["page_end"],
            section=metadata["section"],
            score=1.0 - distance,
        )
        for chunk_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        )
    ]
