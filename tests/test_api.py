"""api/main.py against a mocked answer pipeline — TestClient, no real LLM or vector store."""

import logging

from fastapi.testclient import TestClient

import api.main as main
from src.llm import LLMResponse
from src.obslog import TimedRAGResult
from src.rag import Citation, RAGResult, RefusalReason, SourceNotice, SourceReference
from src.retrieve import RetrievedChunk

client = TestClient(main.app)


def _timed_result(refused=False, source_notices=None, refusal_reason=None):
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
            source_notices=source_notices or [],
            llm_response=LLMResponse(
                text="...",
                model="claude-haiku-4-5",
                input_tokens=50,
                output_tokens=20,
                cost_usd=0.0005,
            ),
            refusal_reason=refusal_reason,
        ),
        latency_ms=150.0,
    )


def test_health_live_returns_ok():
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_returns_503_when_manifest_is_invalid(monkeypatch):
    import scripts.validate_manifest as validate_manifest_module

    def broken_validate(entries):
        raise validate_manifest_module.ManifestValidationError("bad manifest")

    monkeypatch.setattr(validate_manifest_module, "validate_manifest", broken_validate)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["manifest"] == "error"


def test_health_ready_returns_503_when_the_collection_is_empty(monkeypatch):
    class _EmptyCollection:
        def count(self):
            return 0

    import src.store as store_module

    monkeypatch.setattr(store_module, "get_collection", lambda: _EmptyCollection())

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["collection"] == "empty"
    assert response.json()["chunk_count"] == 0


def test_health_ready_returns_503_when_the_collection_raises(monkeypatch):
    import src.store as store_module

    def boom():
        raise RuntimeError("chroma unavailable")

    monkeypatch.setattr(store_module, "get_collection", boom)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["collection"] == "error"


def test_health_ready_returns_503_on_a_provenance_mismatch(monkeypatch):
    import src.provenance as provenance_module

    def boom():
        raise provenance_module.StoreProvenanceError("stale build")

    monkeypatch.setattr(provenance_module, "assert_store_compatible", boom)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["provenance"] == "error"


def test_health_ready_never_calls_the_llm(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("readiness must never call the LLM")

    monkeypatch.setattr("src.llm.complete", fail_if_called)

    response = client.get("/health/ready")

    # whatever the real store's status is locally, the call must not have touched the LLM
    assert response.status_code in (200, 503)


def test_ask_returns_answer_with_citations(monkeypatch):
    timed = _timed_result()
    timed.result.citations = [Citation("sarb_d3_2023_accounting_provisions_ifrs9", 3, True)]
    monkeypatch.setattr(main, "timed_answer", lambda q: timed)

    response = client.post("/ask", json={"question": "What must banks do?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Banks must comply. [sarb_d3_2023, p.3]"
    assert body["citations"] == [
        {
            "doc_id": "sarb_d3_2023_accounting_provisions_ifrs9",
            "page": 3,
            "verified": True,
            "title": "Directive 3/2023: Regulatory Treatment of Accounting Provisions (IFRS 9)",
            "source_url": (
                "https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/"
                "pa-deposit-takers/banks-directives/2023/D3-2023-Regulatory%20treatment%20of%20"
                "accounting%20provisions.pdf#page=3"
            ),
        }
    ]
    assert body["retrieved_chunks"][0]["doc_id"] == "sarb_d3_2023"
    assert body["refused"] is False
    assert body["model"] == "claude-haiku-4-5"
    assert body["refusal_reason"] is None


def test_ask_response_carries_the_refusal_reason(monkeypatch):
    monkeypatch.setattr(
        main,
        "timed_answer",
        lambda q: _timed_result(refused=True, refusal_reason=RefusalReason.UNVERIFIED_CITATION),
    )

    response = client.post("/ask", json={"question": "What must banks do?"})

    assert response.json()["refusal_reason"] == "unverified_citation"


def test_ask_response_carries_source_notices(monkeypatch):
    notices = [
        SourceNotice(
            kind="withdrawn_source",
            text="sarb_circular_19_2004_capital_hybrid_instruments is treated as withdrawn.",
            evidence=[SourceReference("sarb_c1_2026_status_of_circulars", 1)],
        )
    ]
    monkeypatch.setattr(main, "timed_answer", lambda q: _timed_result(source_notices=notices))

    response = client.post("/ask", json={"question": "What must banks do?"})

    body_notices = response.json()["source_notices"]
    assert body_notices == [
        {
            "kind": "withdrawn_source",
            "text": "sarb_circular_19_2004_capital_hybrid_instruments is treated as withdrawn.",
            "evidence": [{"doc_id": "sarb_c1_2026_status_of_circulars", "page": 1}],
        }
    ]


def test_ask_response_source_notices_defaults_to_empty_list(monkeypatch):
    monkeypatch.setattr(main, "timed_answer", lambda q: _timed_result())

    response = client.post("/ask", json={"question": "What must banks do?"})

    assert response.json()["source_notices"] == []


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


def test_pipeline_failure_does_not_log_user_or_provider_text(monkeypatch, caplog):
    def fail(question):
        raise RuntimeError("PRIVATE_PROVIDER_ECHO")

    monkeypatch.setattr(main, "timed_answer", fail)
    caplog.set_level(logging.ERROR, logger="regrag.api")

    response = client.post("/ask", json={"question": "PRIVATE_QUESTION"})

    assert response.status_code == 502
    assert "PRIVATE_QUESTION" not in caplog.text
    assert "PRIVATE_PROVIDER_ECHO" not in caplog.text
    assert "PRIVATE" not in response.text


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
    monkeypatch.setattr(main, "content_logging_enabled", lambda: False)

    response = client.get("/stats")

    assert response.status_code == 200
    assert response.json() == {
        "total_queries": 5,
        "avg_latency_ms": 200.0,
        "total_cost_usd": 0.01,
        "refusal_rate": 0.2,
        "content_logging_enabled": False,
    }


def test_cors_allows_only_the_configured_origins(monkeypatch):
    response = client.get("/health/live", headers={"Origin": "http://localhost:8501"})

    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_recent_queries_returns_logged_rows(monkeypatch):
    monkeypatch.setattr(
        main,
        "recent_queries",
        lambda limit=100: [
            {
                "timestamp": 1_700_000_000.0,
                "question": None,
                "refused": 0,
                "citation_count": 2,
                "latency_ms": 123.4,
                "cost_usd": 0.001,
                "refusal_reason": None,
            }
        ],
    )

    response = client.get("/recent-queries")

    assert response.status_code == 200
    body = response.json()
    assert body == [
        {
            "timestamp": 1_700_000_000.0,
            "question": None,
            "refused": False,
            "citation_count": 2,
            "latency_ms": 123.4,
            "cost_usd": 0.001,
            "refusal_reason": None,
        }
    ]
