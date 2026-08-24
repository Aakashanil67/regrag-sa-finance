"""SQLite log of every query: question, retrieved chunk ids, answer, latency, tokens, cost.

This is the raw material api/main.py's /stats endpoint and app/ops.py's usage charts read from —
one row per question, not aggregated, so both can compute whatever slice they need (latency
percentiles, cost per day, refusal rate) without re-running the RAG pipeline.
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass

from src.config import ROOT
from src.rag import RAGResult

DB_PATH = ROOT / "regrag_log.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    refused INTEGER NOT NULL,
    citation_count INTEGER NOT NULL,
    unverified_citation_count INTEGER NOT NULL,
    retrieved_chunk_ids TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@dataclass
class TimedRAGResult:
    """Wraps a RAGResult with the wall-clock time answer_question() actually took — rag.py
    itself doesn't measure latency, since that's an observability concern, not a RAG one."""

    result: RAGResult
    latency_ms: float


def log_query(timed: TimedRAGResult) -> None:
    result = timed.result
    unverified = sum(1 for c in result.citations if not c.verified)

    with _connect() as conn:
        conn.execute(
            "INSERT INTO queries (timestamp, question, answer, refused, citation_count, "
            "unverified_citation_count, retrieved_chunk_ids, model, input_tokens, output_tokens, "
            "cost_usd, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                result.question,
                result.answer,
                int(result.refused),
                len(result.citations),
                unverified,
                json.dumps([c.chunk_id for c in result.retrieved_chunks]),
                result.llm_response.model,
                result.llm_response.input_tokens,
                result.llm_response.output_tokens,
                result.llm_response.cost_usd,
                timed.latency_ms,
            ),
        )


def timed_answer(question: str, k: int = 5) -> TimedRAGResult:
    from src.rag import answer_question

    start = time.perf_counter()
    result = answer_question(question, k=k)
    latency_ms = (time.perf_counter() - start) * 1000
    timed = TimedRAGResult(result=result, latency_ms=latency_ms)
    log_query(timed)
    return timed


def recent_queries(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM queries ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def stats_summary() -> dict:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n, AVG(latency_ms) AS avg_latency_ms, "
            "SUM(cost_usd) AS total_cost_usd, SUM(refused) AS refused_count "
            "FROM queries"
        ).fetchone()
    n = row[0] or 0
    return {
        "total_queries": n,
        "avg_latency_ms": row[1] or 0.0,
        "total_cost_usd": row[2] or 0.0,
        "refusal_rate": (row[3] or 0) / n if n else 0.0,
    }
