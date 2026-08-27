"""Exact-match response cache, keyed by a hash of the normalised question text.

Deliberately exact-match, not semantic: two questions with the same meaning but different wording
("what must banks do about X" vs "what do banks need to do regarding X") are cache misses. A
semantic cache (embed the query, check similarity against past queries) would catch more repeats,
but risks serving a cached answer to a question that's subtly different from the one that was
actually asked — wrong for a tool whose whole premise is citation accuracy. Exact-match trades
hit rate for that guarantee.

A cache hit skips retrieval and the LLM call entirely, so it doesn't repopulate
`RAGResult.retrieved_chunks` — the "what was retrieved" debug view is empty on a cache hit and
`reports/perf.md` says so; reconstructing it would mean re-running retrieval anyway, which
defeats the point of caching latency.

The key is a hash of the question *and* a fingerprint of the configuration that produced the
answer. Question text alone was the first version and it was wrong in a way that only shows up
across a config change: adopting chunk_size=800 + reranking rebuilt the vector store and changed
what retrieval returns, but every previously cached answer kept its key and kept being served, so
the running product would have gone on returning pre-improvement answers indefinitely while the
eval reports described the new behaviour. Anything that changes what an answer would be — chunk
size, either model, the collection, the system prompt — has to change the key, or the cache is a
silent source of stale output. Old-fingerprint rows simply stop being reachable, which is the
intended invalidation; they are dead weight in the file, not a correctness problem.

Note that `flagged_injection` is deliberately *not* stored here. It's a property of the question
text, not of the answer, so `obslog.timed_answer` recomputes it on every call including cache
hits — see the comment there for what went wrong when this was left to the cache.

`source_notices` (third-party-source and dated-instrument disclosures, see rag.py) isn't stored
either, for a related but distinct reason: it's a pure function of *which documents got cited*,
already stored as `citations_json`, so persisting a second, derived copy would just be another
value that can drift from its source if `rag._source_notices`'s logic ever changes — recomputed on
every read instead, cache hit or not.
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager

from src.config import (
    CACHE_DB_PATH,
    CHUNK_TARGET_TOKENS,
    COLLECTION_NAME,
    CROSS_ENCODER_MODEL_NAME,
    DEFAULT_ANTHROPIC_MODEL,
    EMBEDDING_MODEL_NAME,
)
from src.llm import LLMResponse
from src.rag import _SYSTEM_PROMPT, Citation, RAGResult, _source_notices

_SCHEMA = """
CREATE TABLE IF NOT EXISTS response_cache (
    question_hash TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    refused INTEGER NOT NULL,
    model TEXT NOT NULL
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(_SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _normalize(question: str) -> str:
    return " ".join(question.strip().lower().split())


def _config_fingerprint() -> str:
    """Everything that changes what an answer would be, collapsed to one short hash. The system
    prompt is included by content rather than by version number so an edit to a citation or
    refusal rule invalidates the cache on its own, without anyone remembering to bump a counter."""
    parts = [
        EMBEDDING_MODEL_NAME,
        CROSS_ENCODER_MODEL_NAME,
        COLLECTION_NAME,
        str(CHUNK_TARGET_TOKENS),
        DEFAULT_ANTHROPIC_MODEL,
        hashlib.sha256(_SYSTEM_PROMPT.encode()).hexdigest()[:16],
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _question_hash(question: str) -> str:
    return hashlib.sha256(f"{_config_fingerprint()}:{_normalize(question)}".encode()).hexdigest()


def get_cached(question: str) -> RAGResult | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT question, answer, citations_json, refused, model FROM response_cache "
            "WHERE question_hash = ?",
            (_question_hash(question),),
        ).fetchone()

    if row is None:
        return None

    _, answer, citations_json, refused, model = row
    citations = [Citation(**c) for c in json.loads(citations_json)]
    return RAGResult(
        question=question,
        answer=answer,
        citations=citations,
        retrieved_chunks=[],
        refused=bool(refused),
        flagged_injection=False,
        source_notices=[] if refused else _source_notices(citations),
        llm_response=LLMResponse(
            text=answer, model=model, input_tokens=0, output_tokens=0, cost_usd=0.0
        ),
    )


def set_cached(question: str, result: RAGResult) -> None:
    citations_json = json.dumps(
        [{"doc_id": c.doc_id, "page": c.page, "verified": c.verified} for c in result.citations]
    )
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO response_cache "
            "(question_hash, question, answer, citations_json, refused, model) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                _question_hash(question),
                question,
                result.answer,
                citations_json,
                int(result.refused),
                result.llm_response.model,
            ),
        )
