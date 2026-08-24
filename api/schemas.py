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


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    refused: bool
    latency_ms: float
    cost_usd: float
    model: str


class HealthResponse(BaseModel):
    status: str


class StatsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    total_cost_usd: float
    refusal_rate: float
