"""SQLite log of every query: metrics, refusal reason, retrieved chunk ids, latency, tokens, cost.

One row per question, unaggregated, so api/main.py's /stats and app/ops.py can slice it freely.
Privacy: `LOG_RAW_CONTENT` (default false) gates whether question/answer text is stored at all,
because people paste real account details into compliance tools. With `LOG_HASH_KEY` set,
`question_hmac` stores an HMAC-SHA256 of the normalised question to spot repeats. That is
pseudonymous, not anonymous: anyone with the key can test guesses.

`LOG_RAW_MODEL_OUTPUT` (default false) also keeps the model's pre-validation text, which can hold
a hallucinated citation that was never shown. A cache hit calls no model, so `raw_output_logged`
records that case as off.
"""

import hashlib
import hmac as hmac_module
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass

from src.config import ROOT
from src.privacy import privacy_settings
from src.rag import RAGResult

DB_PATH = ROOT / "regrag_log.sqlite3"

# (column, full column definition), used for both CREATE TABLE and, for anything added after the
# table already existed on disk, an ALTER TABLE migration. `flagged_injection` and `cache_hit`
# were added in later commits than the original table. Without this, CREATE TABLE IF NOT EXISTS
# silently no-ops against an older on-disk schema and every insert starts failing at runtime
# instead of at startup, the failure mode this project's own log file hit once already.
_COLUMNS = [
    ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("timestamp", "REAL NOT NULL"),
    ("question", "TEXT NULL"),
    ("answer", "TEXT NULL"),
    ("question_hmac", "TEXT NULL"),
    ("content_logged", "INTEGER NOT NULL DEFAULT 0"),
    ("raw_model_output", "TEXT NULL"),
    ("raw_output_logged", "INTEGER NOT NULL DEFAULT 0"),
    ("refused", "INTEGER NOT NULL"),
    ("flagged_injection", "INTEGER NOT NULL DEFAULT 0"),
    ("cache_hit", "INTEGER NOT NULL DEFAULT 0"),
    ("citation_count", "INTEGER NOT NULL"),
    ("unverified_citation_count", "INTEGER NOT NULL"),
    ("retrieved_chunk_ids", "TEXT NOT NULL"),
    ("model", "TEXT NOT NULL"),
    ("input_tokens", "INTEGER NOT NULL"),
    ("output_tokens", "INTEGER NOT NULL"),
    ("cost_usd", "REAL NOT NULL"),
    ("latency_ms", "REAL NOT NULL"),
    ("refusal_reason", "TEXT NULL"),
]

_CREATE_TABLE = (
    f"CREATE TABLE IF NOT EXISTS queries ({', '.join(f'{c} {d}' for c, d in _COLUMNS)});"
)


def _has_not_null_content_columns(conn: sqlite3.Connection) -> bool:
    # PRAGMA table_info row shape: (cid, name, type, notnull, dflt_value, pk)
    info = {row[1]: row[3] for row in conn.execute("PRAGMA table_info(queries)")}
    return info.get("question") == 1 or info.get("answer") == 1


def _migrate_to_nullable_content(conn: sqlite3.Connection) -> None:
    """SQLite can't drop a NOT NULL constraint with ALTER TABLE, the only way to relax `question`/
    `answer` to nullable is to rebuild the table under a new schema and copy the old rows across.
    Existing content is preserved as-is. Nulling it out is a separate, explicit scrub operation
    (`scrub_content`), never an automatic side effect of this migration."""
    old_columns = {row[1] for row in conn.execute("PRAGMA table_info(queries)")}
    conn.execute("ALTER TABLE queries RENAME TO queries_v1")
    conn.execute(_CREATE_TABLE)
    common = [c for c, _ in _COLUMNS if c in old_columns]
    cols_sql = ", ".join(common)
    conn.execute(f"INSERT INTO queries ({cols_sql}) SELECT {cols_sql} FROM queries_v1")
    conn.execute("DROP TABLE queries_v1")


def _migrate(conn: sqlite3.Connection) -> None:
    if _has_not_null_content_columns(conn):
        _migrate_to_nullable_content(conn)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(queries)")}
    for column, definition in _COLUMNS:
        if column not in existing:
            conn.execute(f"ALTER TABLE queries ADD COLUMN {column} {definition}")


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_CREATE_TABLE)
    _migrate(conn)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _normalize(question: str) -> str:
    return " ".join(question.strip().lower().split())


@dataclass
class TimedRAGResult:
    """A pipeline result with elapsed time and cache status."""

    result: RAGResult
    latency_ms: float
    cache_hit: bool = False


def log_query(timed: TimedRAGResult) -> None:
    result = timed.result
    unverified = sum(1 for c in result.citations if not c.verified)

    settings = privacy_settings()
    log_raw = settings.log_raw_content
    log_raw_output = settings.log_raw_model_output
    hash_key = os.environ.get("LOG_HASH_KEY") or None

    question = result.question if log_raw else None
    answer = result.answer if log_raw else None
    question_hmac = (
        hmac_module.new(
            hash_key.encode("utf-8"), _normalize(result.question).encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if hash_key
        else None
    )
    # a cache hit never calls the model, so llm_response.text is the *validated* answer
    # reconstructed by cache.get_cached, not raw model output, storing it under this column would
    # be indistinguishable from a real capture. `or None` folds NO_CONTEXT's empty string to NULL.
    raw_model_output = (
        (result.llm_response.text or None) if (log_raw_output and not timed.cache_hit) else None
    )

    with _connect() as conn:
        conn.execute(
            "INSERT INTO queries (timestamp, question, answer, question_hmac, content_logged, "
            "raw_model_output, raw_output_logged, refused, flagged_injection, cache_hit, "
            "citation_count, unverified_citation_count, retrieved_chunk_ids, model, input_tokens, "
            "output_tokens, cost_usd, latency_ms, refusal_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                question,
                answer,
                question_hmac,
                int(log_raw),
                raw_model_output,
                int(log_raw_output),
                int(result.refused),
                int(result.flagged_injection),
                int(timed.cache_hit),
                len(result.citations),
                unverified,
                json.dumps([c.chunk_id for c in result.retrieved_chunks]),
                result.llm_response.model,
                result.llm_response.input_tokens,
                result.llm_response.output_tokens,
                result.llm_response.cost_usd,
                timed.latency_ms,
                result.refusal_reason.value if result.refusal_reason else None,
            ),
        )


def purge_expired(retention_days: int | None = None, now: float | None = None) -> int:
    """Remove rows older than LOG_RETENTION_DAYS."""
    if retention_days is None:
        retention_days = int(os.environ.get("LOG_RETENTION_DAYS", "30"))
    cutoff = (now if now is not None else time.time()) - retention_days * 86400
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM queries WHERE timestamp < ?", (cutoff,))
        return cursor.rowcount


def scrub_content() -> int:
    """Explicit, one-time removal of raw text from rows that logged it, for a local user who ran
    with LOG_RAW_CONTENT and/or LOG_RAW_MODEL_OUTPUT true and changed their mind. Nulls content
    only. Aggregate metrics (timings, refusal reason, citation counts) are untouched, since those
    were never the privacy concern. Both flags are scrubbed together, in one statement: a row can
    have raw_output_logged=1 with content_logged=0 (LOG_RAW_MODEL_OUTPUT on, LOG_RAW_CONTENT off),
    and scrubbing only rows matching one flag would silently leave the other's text behind."""
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE queries SET question = NULL, answer = NULL, raw_model_output = NULL, "
            "content_logged = 0, raw_output_logged = 0 "
            "WHERE content_logged = 1 OR raw_output_logged = 1"
        )
        return cursor.rowcount


def timed_answer(question: str, k: int = 5) -> TimedRAGResult:
    from src.cache import get_cached, set_cached
    from src.guardrails import contains_injection_attempt
    from src.rag import answer_question

    start = time.perf_counter()

    cached = get_cached(question, k=k)
    if cached is not None:
        # Cache hits skip rag.answer_question, so detect repeated injection attempts here too.
        cached.flagged_injection = contains_injection_attempt(question)
        timed = TimedRAGResult(
            result=cached, latency_ms=(time.perf_counter() - start) * 1000, cache_hit=True
        )
    else:
        result = answer_question(question, k=k)
        timed = TimedRAGResult(
            result=result, latency_ms=(time.perf_counter() - start) * 1000, cache_hit=False
        )
        set_cached(question, result, k=k)

    log_query(timed)
    return timed


def recent_queries(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            # id, not timestamp: two inserts landing in the same time.time() tick (a real
            # occurrence on a fast filesystem/clock) make timestamp DESC an unstable order,
            # id is monotonically increasing and always reflects actual insert order
            "SELECT * FROM queries ORDER BY id DESC LIMIT ?",
            (limit,),
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


def content_logging_enabled() -> bool:
    return privacy_settings().log_raw_content


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scrub-content",
        action="store_true",
        help=(
            "null raw question/answer/model-output text from rows that logged any of it, "
            "keeping aggregate metrics"
        ),
    )
    parser.add_argument(
        "--purge-expired",
        action="store_true",
        help="delete rows older than LOG_RETENTION_DAYS (default 30)",
    )
    args = parser.parse_args()

    if args.scrub_content:
        print(f"scrubbed raw content from {scrub_content()} row(s)")
    elif args.purge_expired:
        print(f"purged {purge_expired()} expired row(s)")
    else:
        print(stats_summary())


if __name__ == "__main__":
    main()
