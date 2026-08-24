"""obslog.py against a throwaway SQLite file — never the real regrag_log.sqlite3."""

import src.cache as cache_module
import src.rag as rag_module
from src import obslog
from src.llm import LLMResponse
from src.obslog import TimedRAGResult, log_query, recent_queries, stats_summary, timed_answer
from src.rag import Citation, RAGResult


def _result(refused=False, citations=None, cost_usd=0.001) -> RAGResult:
    return RAGResult(
        question="What must banks do?",
        answer="Banks must comply. [sarb_d3_2023, p.3]",
        citations=citations if citations is not None else [Citation("sarb_d3_2023", 3, True)],
        retrieved_chunks=[],
        refused=refused,
        flagged_injection=False,
        llm_response=LLMResponse(
            text="...",
            model="claude-haiku-4-5",
            input_tokens=100,
            output_tokens=50,
            cost_usd=cost_usd,
        ),
    )


def test_log_query_then_stats_summary_reflects_one_query(tmp_path, monkeypatch):
    monkeypatch.setattr(obslog, "DB_PATH", tmp_path / "test_log.sqlite3")

    log_query(TimedRAGResult(result=_result(), latency_ms=250.0))
    stats = stats_summary()

    assert stats["total_queries"] == 1
    assert stats["avg_latency_ms"] == 250.0
    assert stats["total_cost_usd"] == 0.001
    assert stats["refusal_rate"] == 0.0


def test_refusal_rate_counts_refused_queries(tmp_path, monkeypatch):
    monkeypatch.setattr(obslog, "DB_PATH", tmp_path / "test_log.sqlite3")

    log_query(TimedRAGResult(result=_result(refused=False), latency_ms=100.0))
    log_query(TimedRAGResult(result=_result(refused=True, citations=[]), latency_ms=100.0))

    stats = stats_summary()

    assert stats["total_queries"] == 2
    assert stats["refusal_rate"] == 0.5


def test_stats_summary_on_empty_log_does_not_divide_by_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(obslog, "DB_PATH", tmp_path / "empty_log.sqlite3")

    stats = stats_summary()

    assert stats == {
        "total_queries": 0,
        "avg_latency_ms": 0.0,
        "total_cost_usd": 0.0,
        "refusal_rate": 0.0,
    }


def test_timed_answer_skips_the_rag_pipeline_on_a_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(obslog, "DB_PATH", tmp_path / "test_log.sqlite3")
    monkeypatch.setattr(cache_module, "get_cached", lambda q: _result())

    def fail_if_called(*args, **kwargs):
        raise AssertionError("answer_question() should not run on a cache hit")

    monkeypatch.setattr(rag_module, "answer_question", fail_if_called)

    timed = timed_answer("What must banks do?")

    assert timed.cache_hit is True
    assert recent_queries(limit=1)[0]["cache_hit"] == 1


def test_timed_answer_populates_the_cache_on_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(obslog, "DB_PATH", tmp_path / "test_log.sqlite3")
    monkeypatch.setattr(cache_module, "get_cached", lambda q: None)
    monkeypatch.setattr(rag_module, "answer_question", lambda q, k=5: _result())

    stored = {}
    monkeypatch.setattr(cache_module, "set_cached", lambda q, r: stored.setdefault("result", r))

    timed = timed_answer("What must banks do?")

    assert timed.cache_hit is False
    assert stored["result"].answer == _result().answer
