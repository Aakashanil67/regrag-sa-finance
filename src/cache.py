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
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager

from src.config import CACHE_DB_PATH
from src.llm import LLMResponse
from src.rag import Citation, RAGResult

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


def _question_hash(question: str) -> str:
    return hashlib.sha256(_normalize(question).encode()).hexdigest()


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
