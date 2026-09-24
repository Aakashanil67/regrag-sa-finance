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

The `bm25` and `hybrid` strategies add a lexical ranking so identifiers such as "Directive 8/2025"
are matched exactly; `hybrid` fuses it with the dense ranking by reciprocal rank fusion.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.config import (
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_MODEL_REVISION,
    MANIFEST_PATH,
    RERANK_CANDIDATE_POOL_SIZE,
    RETRIEVAL_STRATEGY,
)
from src.store import embed_texts, get_collection

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        import os

        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(
            CROSS_ENCODER_MODEL_NAME,
            revision=os.environ.get("RERANKER_MODEL_REVISION", CROSS_ENCODER_MODEL_REVISION),
        )
        from src.provenance import assert_loaded_model_revision

        assert_loaded_model_revision(_cross_encoder, "reranker")
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


class RetrievedChunks(list[RetrievedChunk]):
    """List-compatible retrieval output with named-source coverage metadata."""

    def __init__(self, chunks=(), *, coverage=None):
        super().__init__(chunks)
        self.coverage = coverage or {
            "requested": [],
            "resolved": [],
            "represented": list(dict.fromkeys(chunk.doc_id for chunk in self)),
            "incomplete": False,
        }


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


def _normalise_name(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _entry_aliases(entry: dict) -> set[str]:
    title = entry.get("title", "")
    aliases = {entry["id"], title}
    aliases.add(title.split(":", 1)[0])
    title_without_parenthetical = re.sub(r"\s*\([^)]*\)", "", title)
    title_without_year = re.sub(r"\b(?:19|20)\d{2}\b", "", title_without_parenthetical)
    aliases.update(
        {
            title_without_parenthetical,
            title_without_year,
            title_without_parenthetical.split(":", 1)[-1],
            title_without_year.split(":", 1)[-1],
            re.sub("over(?:-the-| )counter", "OTC", title_without_year, flags=re.IGNORECASE),
        }
    )
    year_match = re.search(r"\b(19|20)\d{2}\b", title)
    subject_match = re.search(r"\b(IFRS\s+\d+)\b", title, re.IGNORECASE)
    if year_match and subject_match and "issued text" in title.lower():
        aliases.add(f"{year_match.group(0)} issued {subject_match.group(1)} text")
    for match in re.finditer(
        r"\b(directive|guidance\s+note|guideline|circular)\s+([dgc]?)\s*(\d{1,3})\s*(?:/|of)\s*(\d{4})\b",
        title,
        re.IGNORECASE,
    ):
        kind, prefix, number, year = match.groups()
        kind = kind.lower().replace("  ", " ")
        aliases.update(
            {
                f"{kind} {number}/{year}",
                f"{kind} {prefix}{number}/{year}",
                f"{prefix or kind[0]}{number}/{year}",
            }
        )
    return {_normalise_name(alias) for alias in aliases if alias}


def resolve_named_documents(query: str, manifest: list[dict]) -> list[str]:
    """Resolve explicit, manifest-backed document names without guessing bare numbers.

    Aliases are only accepted when they map to one manifest entry. This intentionally leaves a
    query such as ``3/2020`` unresolved: a number/year without an instrument type is ambiguous in
    a regulatory corpus and must not silently select a document.
    """
    alias_to_ids: dict[str, set[str]] = {}
    for entry in manifest:
        for alias in _entry_aliases(entry):
            alias_to_ids.setdefault(alias, set()).add(entry["id"])

    normalised_query = _normalise_name(query)
    padded_query = f" {normalised_query} "
    matches: set[str] = set()
    for alias, ids in alias_to_ids.items():
        if len(ids) != 1:
            continue
        if f" {alias} " in padded_query:
            matches.update(ids)

    manifest_order = {entry["id"]: index for index, entry in enumerate(manifest)}
    return sorted(matches, key=lambda doc_id: manifest_order[doc_id])


@lru_cache(maxsize=1)
def _manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _rerank_candidates(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not candidates:
        return candidates
    cross_encoder = _get_cross_encoder()
    pairs = [(query, candidate.text) for candidate in candidates]
    cross_scores = cross_encoder.predict(pairs)
    for chunk, score in zip(candidates, cross_scores, strict=True):
        chunk.score = float(score)
    candidates.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
    return candidates


def _named_balanced(
    query: str,
    *,
    k: int,
    rerank: bool,
    collection,
) -> RetrievedChunks:
    requested = resolve_named_documents(query, _manifest())
    pool_size = max(k, RERANK_CANDIDATE_POOL_SIZE) if rerank else k
    global_candidates = _fetch_candidates(query, pool_size, None, collection=collection)
    named_candidates = (
        _fetch_candidates(query, pool_size, requested, collection=collection) if requested else []
    )

    deduped: dict[str, RetrievedChunk] = {}
    for candidate in global_candidates + named_candidates:
        deduped.setdefault(candidate.chunk_id, candidate)
    candidates = list(deduped.values())
    if rerank:
        _rerank_candidates(query, candidates)
    else:
        candidates.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))

    available_named = [
        doc_id for doc_id in requested if any(chunk.doc_id == doc_id for chunk in candidates)
    ]
    reserved = []
    reserved_ids = set()
    for doc_id in available_named:
        source_candidates = [chunk for chunk in candidates if chunk.doc_id == doc_id]
        if source_candidates and len(reserved) < k:
            selected = source_candidates[0]
            reserved.append(selected)
            reserved_ids.add(selected.chunk_id)

    remaining = [chunk for chunk in candidates if chunk.chunk_id not in reserved_ids]
    selected = (reserved + remaining)[:k]
    represented = list(dict.fromkeys(chunk.doc_id for chunk in selected))
    return RetrievedChunks(
        selected,
        coverage={
            "requested": requested,
            "resolved": available_named,
            "represented": represented,
            "incomplete": any(doc_id not in represented for doc_id in requested),
        },
    )


_BM25_TOKEN = re.compile(r"[a-z0-9]+(?:[/.][a-z0-9]+)*")
RRF_K = 60  # the usual constant from Cormack et al. (2009); not tuned here
HYBRID_POOL = 50
_bm25_cache: dict[tuple[str, int], tuple] = {}


def _bm25_tokens(text: str) -> list[str]:
    return _BM25_TOKEN.findall(text.lower())


def _bm25_index(collection):
    key = (collection.name, collection.count())
    if key not in _bm25_cache:
        from rank_bm25 import BM25Okapi

        data = collection.get(include=["documents", "metadatas"])
        order = sorted(range(len(data["ids"])), key=lambda i: data["ids"][i])
        ids = [data["ids"][i] for i in order]
        docs = [data["documents"][i] for i in order]
        metas = [data["metadatas"][i] for i in order]
        tokens = [_bm25_tokens(d) for d in docs]
        _bm25_cache[key] = (BM25Okapi(tokens), ids, docs, metas, [set(t) for t in tokens])
    return _bm25_cache[key]


def _bm25_candidates(
    query: str, n: int, collection=None, doc_ids: list[str] | None = None
) -> list[RetrievedChunk]:
    if collection is None:
        collection = get_collection()
    bm25, ids, docs, metas, token_sets = _bm25_index(collection)
    query_tokens = _bm25_tokens(query)
    scores = bm25.get_scores(query_tokens)
    out = []
    for i in np.argsort(-scores, kind="stable"):
        if len(out) == n:
            break
        # a term in over half the chunks gets a negative idf, so "matched" is judged by token
        # overlap rather than by the sign of the score
        if token_sets[i].isdisjoint(query_tokens):
            continue
        if doc_ids and metas[i]["doc_id"] not in doc_ids:
            continue
        out.append(
            RetrievedChunk(
                chunk_id=ids[i],
                doc_id=metas[i]["doc_id"],
                text=docs[i],
                page_start=metas[i]["page_start"],
                page_end=metas[i]["page_end"],
                section=metas[i]["section"],
                score=float(scores[i]),
            )
        )
    return out


def _rrf(ranked_lists: list[list[RetrievedChunk]], n: int) -> list[RetrievedChunk]:
    fused: dict[str, float] = {}
    first: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            first.setdefault(chunk.chunk_id, chunk)
    out = []
    for chunk_id in sorted(fused, key=lambda cid: (-fused[cid], cid))[:n]:
        first[chunk_id].score = fused[chunk_id]
        out.append(first[chunk_id])
    return out


def retrieve(
    query: str,
    k: int = 5,
    doc_ids: list[str] | None = None,
    rerank: bool = False,
    collection=None,
    strategy: str = RETRIEVAL_STRATEGY,
) -> RetrievedChunks:
    if strategy not in {"semantic", "named_balanced", "bm25", "hybrid"}:
        raise ValueError(f"unknown retrieval strategy: {strategy}")
    if strategy == "named_balanced":
        return _named_balanced(query, k=k, rerank=rerank, collection=collection)

    pool_size = max(k, RERANK_CANDIDATE_POOL_SIZE) if rerank else k
    if strategy == "bm25":
        candidates = _bm25_candidates(query, pool_size, collection, doc_ids)
    elif strategy == "hybrid":
        dense = _fetch_candidates(query, HYBRID_POOL, doc_ids, collection=collection)
        sparse = _bm25_candidates(query, HYBRID_POOL, collection, doc_ids)
        candidates = _rrf([dense, sparse], pool_size)
    else:
        candidates = _fetch_candidates(query, pool_size, doc_ids, collection=collection)

    if not rerank or not candidates:
        return RetrievedChunks(candidates[:k])

    return RetrievedChunks(_rerank_candidates(query, candidates)[:k])


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
