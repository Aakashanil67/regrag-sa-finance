"""Request/response models for the API — the contract the Streamlit UI and any other client
codes against, kept separate from src/rag.py's internal dataclasses so a change to the RAG
pipeline's internals doesn't silently change the wire format."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class CitationOut(BaseModel):
    doc_id: str
    page: int
    verified: bool
    title: str | None = None
    source_url: str | None = None


class RetrievedChunkOut(BaseModel):
    doc_id: str
    page_start: int
    page_end: int
    section: str
    text: str
    score: float


class SourceReferenceOut(BaseModel):
    doc_id: str
    page: int


class SourceNoticeOut(BaseModel):
    kind: str
    text: str
    evidence: list[SourceReferenceOut] = []


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieved_chunks: list[RetrievedChunkOut]
    refused: bool
    refusal_reason: str | None = None
    latency_ms: float
    cost_usd: float
    model: str
    source_notices: list[SourceNoticeOut] = []


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    ready: bool
    manifest: str  # "ok" | "error"
    collection: str  # "ok" | "empty" | "error"
    chunk_count: int
    provenance: str  # "ok" | "error"


class StatsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    total_cost_usd: float
    refusal_rate: float
    content_logging_enabled: bool


class RecentQueryOut(BaseModel):
    timestamp: float
    question: str | None
    refused: bool
    citation_count: int
    latency_ms: float
    cost_usd: float
    refusal_reason: str | None
