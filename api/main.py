"""FastAPI backend: POST /ask, GET /health/live, GET /health/ready, GET /stats.

Every /ask call goes through obslog.timed_answer(), so it's logged to SQLite by construction —
there's no code path that answers a question without also recording it, which is what lets
/stats and the ops dashboard trust the log as a complete picture of usage rather than a sample.
"""

import json
import logging
import os
import uuid
from functools import lru_cache
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    HealthResponse,
    ReadinessResponse,
    RecentQueryOut,
    RetrievedChunkOut,
    SourceNoticeOut,
    SourceReferenceOut,
    StatsResponse,
)
from src.config import MANIFEST_PATH
from src.obslog import content_logging_enabled, recent_queries, stats_summary, timed_answer

logger = logging.getLogger("regrag.api")


@lru_cache(maxsize=1)
def _manifest_records() -> dict[str, dict]:
    """Return curated manifest records used to label and link served citations."""
    try:
        entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {entry["id"]: entry for entry in entries}


def _citation_source_url(record: dict, page: int) -> str | None:
    url = record.get("download_url") or record.get("landing_page_url")
    if not url:
        return None
    parsed = urlsplit(url)
    if parsed.path.lower().endswith(".pdf"):
        return f"{url}#page={page}"
    return url


def _citation_out(citation) -> CitationOut:
    record = _manifest_records().get(citation.doc_id, {})
    return CitationOut(
        doc_id=citation.doc_id,
        page=citation.page,
        verified=citation.verified,
        title=record.get("title"),
        source_url=_citation_source_url(record, citation.page),
        section_ref=citation.section_ref,
    )


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="RegRAG API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# No authentication anywhere in this stack — CORS is the only thing standing between this API and
# any page in the user's browser that decides to call it. Defaults to the local chat origin only;
# widen via CORS_ALLOWED_ORIGINS (comma-separated) for a different local setup, never for a
# public deployment (see README's deployment-boundary section — this stack must stay loopback-only).
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health/live", response_model=HealthResponse)
def health_live() -> HealthResponse:
    """Proves the process is running and can serve a response — nothing more. No file I/O, no
    Chroma, no model. A readiness failure must never make this fail too, or an orchestrator that
    only checks liveness would restart a container that's merely waiting on the vector store."""
    return HealthResponse(status="ok")


@app.get("/health/ready", response_model=None)
def health_ready() -> ReadinessResponse | JSONResponse:
    """Checks manifest validity, Chroma availability, a non-empty collection, and that the active
    pipeline's provenance matches what the store was built from — deliberately never a paid LLM
    call, since readiness is meant to be cheap and pollable, not something that racks up API cost
    every time an orchestrator checks it."""
    from scripts.validate_manifest import ManifestValidationError, validate_manifest
    from src.config import MANIFEST_PATH
    from src.provenance import assert_store_compatible
    from src.store import get_collection

    manifest_status = "ok"
    try:
        validate_manifest(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    except (ManifestValidationError, OSError, ValueError):
        manifest_status = "error"

    collection_status = "error"
    chunk_count = 0
    try:
        chunk_count = get_collection().count()
        collection_status = "ok" if chunk_count > 0 else "empty"
    except Exception:
        collection_status = "error"

    provenance_status = "ok"
    try:
        assert_store_compatible()
    except Exception:  # StoreProvenanceError, or the collection check above already failing
        provenance_status = "error"

    ready = manifest_status == "ok" and collection_status == "ok" and provenance_status == "ok"
    body = ReadinessResponse(
        ready=ready,
        manifest=manifest_status,
        collection=collection_status,
        chunk_count=chunk_count,
        provenance=provenance_status,
    )
    return body if ready else JSONResponse(status_code=503, content=body.model_dump())


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return StatsResponse(**stats_summary(), content_logging_enabled=content_logging_enabled())


@app.get("/recent-queries", response_model=list[RecentQueryOut])
def recent_queries_endpoint(limit: int = 100) -> list[RecentQueryOut]:
    """The ops dashboard's only path to the query log — it must never import src.obslog or open
    the SQLite file directly, so the API stays the one place that decides what's safe to surface
    (this is already respected upstream: `question` is null here whenever LOG_RAW_CONTENT is off,
    since that's what's actually stored)."""
    return [
        RecentQueryOut(
            timestamp=row["timestamp"],
            question=row["question"],
            refused=bool(row["refused"]),
            citation_count=row["citation_count"],
            latency_ms=row["latency_ms"],
            cost_usd=row["cost_usd"],
            refusal_reason=row["refusal_reason"],
        )
        for row in recent_queries(limit=limit)
    ]


@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
def ask(request: Request, body: AskRequest) -> AskResponse | JSONResponse:
    try:
        timed = timed_answer(body.question)
    except Exception as exc:
        # the HTTP boundary: anything from here down (Anthropic API error, chroma I/O, a bad
        # regex) becomes a clean 502 instead of a raw traceback reaching the client. Logged, not
        # swallowed — inner code still raises specific exceptions where it can act on them.
        request_id = uuid.uuid4().hex
        logger.error(
            "answer_question failed request_id=%s error_type=%s",
            request_id,
            type(exc).__name__,
            exc_info=False,
        )
        return JSONResponse(
            status_code=502,
            content={"detail": "The assistant is temporarily unavailable. Please try again."},
        )

    result = timed.result
    return AskResponse(
        answer=result.answer,
        citations=[_citation_out(citation) for citation in result.citations],
        retrieved_chunks=[
            RetrievedChunkOut(
                doc_id=chunk.doc_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                text=chunk.text,
                score=chunk.score,
            )
            for chunk in result.retrieved_chunks
        ],
        refused=result.refused,
        refusal_reason=result.refusal_reason.value if result.refusal_reason else None,
        latency_ms=timed.latency_ms,
        cost_usd=result.llm_response.cost_usd,
        model=result.llm_response.model,
        source_notices=[
            SourceNoticeOut(
                kind=notice.kind,
                text=notice.text,
                evidence=[
                    SourceReferenceOut(doc_id=ref.doc_id, page=ref.page) for ref in notice.evidence
                ],
            )
            for notice in result.source_notices
        ],
    )
