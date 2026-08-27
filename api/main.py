"""FastAPI backend: POST /ask, GET /health, GET /stats.

Every /ask call goes through obslog.timed_answer(), so it's logged to SQLite by construction —
there's no code path that answers a question without also recording it, which is what lets
/stats and the ops dashboard trust the log as a complete picture of usage rather than a sample.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    HealthResponse,
    RetrievedChunkOut,
    StatsResponse,
)
from src.obslog import stats_summary, timed_answer

logger = logging.getLogger("regrag.api")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="RegRAG API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return StatsResponse(**stats_summary())


@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
def ask(request: Request, body: AskRequest) -> AskResponse | JSONResponse:
    try:
        timed = timed_answer(body.question)
    except Exception:
        # the HTTP boundary: anything from here down (Anthropic API error, chroma I/O, a bad
        # regex) becomes a clean 502 instead of a raw traceback reaching the client. Logged, not
        # swallowed — inner code still raises specific exceptions where it can act on them.
        logger.exception("answer_question failed for question=%r", body.question)
        return JSONResponse(
            status_code=502,
            content={"detail": "The assistant is temporarily unavailable. Please try again."},
        )

    result = timed.result
    return AskResponse(
        answer=result.answer,
        citations=[
            CitationOut(doc_id=c.doc_id, page=c.page, verified=c.verified) for c in result.citations
        ],
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
        latency_ms=timed.latency_ms,
        cost_usd=result.llm_response.cost_usd,
        model=result.llm_response.model,
    )
