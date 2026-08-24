"""api/main.py against a mocked answer pipeline — TestClient, no real LLM or vector store."""

from fastapi.testclient import TestClient

import api.main as main
from src.llm import LLMResponse
from src.obslog import TimedRAGResult
from src.rag import Citation, RAGResult
from src.retrieve import RetrievedChunk

client = TestClient(main.app)


def _timed_result(refused=False):
    return TimedRAGResult(
        result=RAGResult(
            question="What must banks do?",
            answer="Banks must comply. [sarb_d3_2023, p.3]",
            citations=[] if refused else [Citation("sarb_d3_2023", 3, True)],
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="hash1",
                    doc_id="sarb_d3_2023",
                    text="Banks must comply with the directive.",
                    page_start=3,
                    page_end=3,
                    section="Executive summary",
                    score=0.87,
                )
            ],
            refused=refused,
            flagged_injection=False,
            llm_response=LLMResponse(
                text="...",
                model="claude-haiku-4-5",
                input_tokens=50,
                output_tokens=20,
                cost_usd=0.0005,
            ),
        ),
        latency_ms=150.0,
    )


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_with_citations(monkeypatch):
    monkeypatch.setattr(main, "timed_answer", lambda q: _timed_result())

    response = client.post("/ask", json={"question": "What must banks do?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Banks must comply. [sarb_d3_2023, p.3]"
    assert body["citations"] == [{"doc_id": "sarb_d3_2023", "page": 3, "verified": True}]
    assert body["retrieved_chunks"][0]["doc_id"] == "sarb_d3_2023"
    assert body["refused"] is False
    assert body["model"] == "claude-haiku-4-5"


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_rejects_oversized_question():
    response = client.post("/ask", json={"question": "x" * 5000})

    assert response.status_code == 422


def test_ask_returns_502_when_pipeline_raises(monkeypatch):
    def boom(q):
        raise RuntimeError("provider is down")

    monkeypatch.setattr(main, "timed_answer", boom)

    response = client.post("/ask", json={"question": "What must banks do?"})

    assert response.status_code == 502
    assert "temporarily unavailable" in response.json()["detail"]


def test_stats_returns_summary(monkeypatch):
    monkeypatch.setattr(
        main,
        "stats_summary",
        lambda: {
            "total_queries": 5,
            "avg_latency_ms": 200.0,
            "total_cost_usd": 0.01,
            "refusal_rate": 0.2,
        },
    )

    response = client.get("/stats")

    assert response.status_code == 200
    assert response.json() == {
        "total_queries": 5,
        "avg_latency_ms": 200.0,
        "total_cost_usd": 0.01,
        "refusal_rate": 0.2,
    }
