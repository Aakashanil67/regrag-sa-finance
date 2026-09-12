"""Top-k semantic retrieval over the chunk store, with optional metadata filters and an optional
cross-encoder reranking pass.

Candidate search is exact cosine similarity computed in-process, not ChromaDB's approximate HNSW
`.query()`. ChromaDB's local HNSW segment rebuilds its graph by re-inserting every embedding on
each fresh process (no persisted index file), using a thread pool sized to the machine's CPU count
by default — parallel insertion order isn't fixed, so the same on-disk data produced a
structurally different graph, and therefore different top-k results, across separate process
launches of this same, unchanged pipeline. That's incompatible with a sealed, run-once release
protocol, which requires identical input to produce identical output. At this corpus's size
(low thousands of chunks), brute-force cosine similarity over every chunk costs low milliseconds —
cheap enough that the "approximate" in approximate nearest neighbour buys nothing here and only
costs reproducibility. This is a scale trade-off, not a free win: past roughly tens of thousands of
chunks, exact search stops being cheap and an ANN index becomes necessary again — which brings this
exact non-determinism back and needs its own answer at that point (a persisted, deterministically
rebuilt index; a different store; a real seeded index build), not an assumption that this fix still
applies unchanged. `RetrievedChunk.score` is that cosine similarity (-1 to 1, higher is more
relevant) except when reranking is on, where `score` is the cross-encoder's own relevance score
instead (unbounded); "higher is more relevant" still holds either way, which is the property
callers actually depend on.

Reranking works in two stages because the two models are good at different things: the bi-encoder
(sentence-transformers, used for the initial candidate search) embeds the query and every chunk
independently, which is fast enough to search the whole collection but can't compare them
directly against each other. The cross-encoder reads the query and one candidate chunk together
in a single forward pass, which is far more accurate but too slow to run against the whole
collection — so it only reranks the bi-encoder's top `RERANK_CANDIDATE_POOL_SIZE` candidates, not
everything.
"""

from dataclasses import dataclass

import numpy as np

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
    query_embedding = np.asarray(embed_texts([query])[0], dtype=np.float64)

    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    results = collection.get(where=where, include=["documents", "metadatas", "embeddings"])

    if not results["ids"]:
        return []

    embeddings = np.asarray(results["embeddings"], dtype=np.float64)
    query_unit = query_embedding / np.linalg.norm(query_embedding)
    doc_units = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = doc_units @ query_unit

    # stable sort: ties keep collection order rather than whatever order argpartition happens to
    # produce, so results are reproducible even when two chunks are exactly equidistant
    order = np.argsort(-similarities, kind="stable")[:n]

    return [
        RetrievedChunk(
            chunk_id=results["ids"][i],
            doc_id=results["metadatas"][i]["doc_id"],
            text=results["documents"][i],
            page_start=results["metadatas"][i]["page_start"],
            page_end=results["metadatas"][i]["page_end"],
            section=results["metadatas"][i]["section"],
            score=float(similarities[i]),
        )
        for i in order
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


def fetch_document_page(doc_id: str, page: int, collection=None) -> list[RetrievedChunk]:
    """Exact, deterministic lookup of every chunk covering one page of one document — used to
    attach status-evidence text (e.g. the page of a circular naming it withdrawn) to generation
    context, independent of whatever the semantic top-k search would have surfaced for the
    question actually asked. Chunks are sorted by page then chunk_id so multiple chunks covering
    the same page return in a stable order rather than whatever order Chroma's `.get()` gives back.
    """
    if collection is None:
        collection = get_collection()

    results = collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])

    chunks = [
        RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=metadata["doc_id"],
            text=document,
            page_start=metadata["page_start"],
            page_end=metadata["page_end"],
            section=metadata["section"],
            score=1.0,
        )
        for chunk_id, document, metadata in zip(
            results["ids"], results["documents"], results["metadatas"], strict=True
        )
        if metadata["page_start"] <= page <= metadata["page_end"]
    ]
    chunks.sort(key=lambda c: (c.page_start, c.chunk_id))
    return chunks
