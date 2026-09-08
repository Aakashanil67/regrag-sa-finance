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

from src.config import CACHE_DB_PATH
from src.llm import LLMResponse
from src.provenance import pipeline_fingerprint
from src.rag import Citation, RAGResult, RefusalReason, _source_notices

_COLUMNS = [
    ("question_hash", "TEXT PRIMARY KEY"),
    ("question", "TEXT NOT NULL"),
    ("answer", "TEXT NOT NULL"),
    ("citations_json", "TEXT NOT NULL"),
    ("refused", "INTEGER NOT NULL"),
    ("model", "TEXT NOT NULL"),
    ("refusal_reason", "TEXT NULL"),
]

_CREATE_TABLE = (
    f"CREATE TABLE IF NOT EXISTS response_cache ({', '.join(f'{c} {d}' for c, d in _COLUMNS)});"
)


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(response_cache)")}
    for column, definition in _COLUMNS:
        if column not in existing:
            conn.execute(f"ALTER TABLE response_cache ADD COLUMN {column} {definition}")


@contextmanager
def _connect():
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(_CREATE_TABLE)
    _migrate(conn)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _normalize(question: str) -> str:
    return " ".join(question.strip().lower().split())


def _question_hash(question: str, *, k: int) -> str:
    # provenance.pipeline_fingerprint is the single source of truth for "everything that changes
    # what an answer would be" — cache.py used to keep its own smaller, drifting copy of that list
    # (see the module docstring for the incident that caused), which is why this calls it rather
    # than assembling config fields locally.
    fingerprint = pipeline_fingerprint(k=k)
    return hashlib.sha256(f"{fingerprint}:{_normalize(question)}".encode()).hexdigest()


def get_cached(question: str, *, k: int) -> RAGResult | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT question, answer, citations_json, refused, model, refusal_reason "
            "FROM response_cache WHERE question_hash = ?",
            (_question_hash(question, k=k),),
        ).fetchone()

    if row is None:
        return None

    _, answer, citations_json, refused, model, refusal_reason = row
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
        refusal_reason=RefusalReason(refusal_reason) if refusal_reason else None,
    )


def set_cached(question: str, result: RAGResult, *, k: int) -> None:
    citations_json = json.dumps(
        [{"doc_id": c.doc_id, "page": c.page, "verified": c.verified} for c in result.citations]
    )
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO response_cache "
            "(question_hash, question, answer, citations_json, refused, model, refusal_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _question_hash(question, k=k),
                question,
                result.answer,
                citations_json,
                int(result.refused),
                result.llm_response.model,
                result.refusal_reason.value if result.refusal_reason else None,
            ),
        )
