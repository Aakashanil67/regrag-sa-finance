"""Top-k semantic retrieval over the chunk store, with optional metadata filters and an optional
cross-encoder reranking pass.

ChromaDB returns cosine distance (0 = identical direction, 2 = opposite); `RetrievedChunk.score`
is `1 - distance` instead, so "higher is more relevant" holds everywhere downstream (the eval
harness, the sources panel, the retrieval benchmark) without every caller re-deriving the sign —
except when reranking is on, where `score` is the cross-encoder's own relevance score instead
(unbounded, not a 0-1 distance-derived value); "higher is more relevant" still holds either way,
which is the property callers actually depend on.

Reranking works in two stages because the two models are good at different things: the bi-encoder
(sentence-transformers, used for the initial Chroma search) embeds the query and every chunk
independently, which is fast enough to search the whole collection but can't compare them
directly against each other. The cross-encoder reads the query and one candidate chunk together
in a single forward pass, which is far more accurate but too slow to run against the whole
collection — so it only reranks the bi-encoder's top `RERANK_CANDIDATE_POOL_SIZE` candidates, not
everything.
"""

from dataclasses import dataclass

from src.config import CROSS_ENCODER_MODEL_NAME, RERANK_CANDIDATE_POOL_SIZE
from src.store import embed_texts, get_collection

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return _cross_encoder


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    section: str
    score: float


def _fetch_candidates(
    query: str, n: int, doc_ids: list[str] | None, collection=None
) -> list[RetrievedChunk]:
    # collection is injectable so scripts/sweep_chunk_size.py can run this exact retrieval logic
    # (reranking included) against a throwaway experimental collection instead of the production
    # one — the alternative, a second hand-rolled copy of this query logic for experiments, risks
    # the experiment silently testing different code than what actually ships.
    if collection is None:
        collection = get_collection()
    query_embedding = embed_texts([query])[0]

    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
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


def retrieve(
    query: str,
    k: int = 5,
    doc_ids: list[str] | None = None,
    rerank: bool = False,
    collection=None,
) -> list[RetrievedChunk]:
    pool_size = max(k, RERANK_CANDIDATE_POOL_SIZE) if rerank else k
    candidates = _fetch_candidates(query, pool_size, doc_ids, collection=collection)

    if not rerank or not candidates:
        return candidates[:k]

    cross_encoder = _get_cross_encoder()
    pairs = [(query, c.text) for c in candidates]
    cross_scores = cross_encoder.predict(pairs)
    for chunk, score in zip(candidates, cross_scores, strict=True):
        chunk.score = float(score)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:k]
