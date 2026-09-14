"""Opt-in exact-match response cache with bounded retention.

The cache is deliberately exact-match, not semantic: a rephrased question is a miss, which avoids
serving an answer to a subtly different request. Persistent storage is disabled by default. When
enabled, only the versioned v2 table is read or written; legacy ``response_cache`` rows remain
untouched until an explicit scrub command.
"""

import argparse
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager

from src.config import CACHE_DB_PATH
from src.llm import LLMResponse
from src.privacy import privacy_settings
from src.provenance import pipeline_fingerprint
from src.rag import Citation, RAGResult, RefusalReason, _source_notices

_V2_COLUMNS = [
    ("question_hash", "TEXT PRIMARY KEY"),
    ("answer", "TEXT NOT NULL"),
    ("citations_json", "TEXT NOT NULL"),
    ("refused", "INTEGER NOT NULL"),
    ("model", "TEXT NOT NULL"),
    ("refusal_reason", "TEXT NULL"),
    ("created_at", "REAL NOT NULL"),
    ("expires_at", "REAL NOT NULL"),
]

_CREATE_V2_TABLE = (
    "CREATE TABLE IF NOT EXISTS response_cache_v2 "
    f"({', '.join(f'{column} {definition}' for column, definition in _V2_COLUMNS)});"
)
_RECOGNISED_TABLES = ("response_cache", "response_cache_v2")


@contextmanager
def _connect():
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(_CREATE_V2_TABLE)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _normalize(question: str) -> str:
    return " ".join(question.strip().lower().split())


def _question_hash(question: str, *, k: int) -> str:
    fingerprint = pipeline_fingerprint(k=k)
    return hashlib.sha256(f"{fingerprint}:{_normalize(question)}".encode()).hexdigest()


def _purge_expired_conn(conn: sqlite3.Connection, *, now: float, dry_run: bool = False) -> int:
    count = conn.execute(
        "SELECT COUNT(*) FROM response_cache_v2 WHERE expires_at <= ?", (now,)
    ).fetchone()[0]
    if count and not dry_run:
        conn.execute("DELETE FROM response_cache_v2 WHERE expires_at <= ?", (now,))
    return count


def get_cached(question: str, *, k: int) -> RAGResult | None:
    settings = privacy_settings()
    if not settings.cache_enabled:
        return None

    now = time.time()
    with _connect() as conn:
        _purge_expired_conn(conn, now=now)
        row = conn.execute(
            "SELECT answer, citations_json, refused, model, refusal_reason "
            "FROM response_cache_v2 WHERE question_hash = ? AND expires_at > ?",
            (_question_hash(question, k=k), now),
        ).fetchone()

    if row is None:
        return None

    answer, citations_json, refused, model, refusal_reason = row
    citations = [Citation(**citation) for citation in json.loads(citations_json)]
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
    settings = privacy_settings()
    if not settings.cache_enabled:
        return

    now = time.time()
    citations_json = json.dumps(
        [{"doc_id": c.doc_id, "page": c.page, "verified": c.verified} for c in result.citations]
    )
    with _connect() as conn:
        _purge_expired_conn(conn, now=now)
        conn.execute(
            "INSERT OR REPLACE INTO response_cache_v2 "
            "(question_hash, answer, citations_json, refused, model, refusal_reason, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _question_hash(question, k=k),
                result.answer,
                citations_json,
                int(result.refused),
                result.llm_response.model,
                result.refusal_reason.value if result.refusal_reason else None,
                now,
                now + settings.cache_ttl_seconds,
            ),
        )


def purge_expired(*, now: float | None = None, dry_run: bool = False) -> int:
    """Count or remove expired v2 rows; disabled caching does not open the database."""
    if not privacy_settings().cache_enabled:
        return 0
    effective_now = time.time() if now is None else now
    with _connect() as conn:
        return _purge_expired_conn(conn, now=effective_now, dry_run=dry_run)


def scrub_cache(*, dry_run: bool = False) -> dict[str, int]:
    """Count or delete rows in recognised legacy and v2 tables.

    This is intentionally independent of ``CACHE_ENABLED``. If no database exists, it returns
    zeroes without creating one.
    """
    counts = dict.fromkeys(_RECOGNISED_TABLES, 0)
    if not CACHE_DB_PATH.exists():
        return counts

    with sqlite3.connect(CACHE_DB_PATH) as conn:
        existing = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
                _RECOGNISED_TABLES,
            )
        }
        for table in _RECOGNISED_TABLES:
            if table not in existing:
                continue
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if counts[table] and not dry_run:
                conn.execute(f"DELETE FROM {table}")
        if not dry_run:
            conn.commit()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--purge-expired", action="store_true")
    group.add_argument("--scrub-content", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.purge_expired:
        print(f"expired={purge_expired(dry_run=args.dry_run)}")
    else:
        counts = scrub_cache(dry_run=args.dry_run)
        print(" ".join(f"{table}={count}" for table, count in counts.items()))


if __name__ == "__main__":
    main()
