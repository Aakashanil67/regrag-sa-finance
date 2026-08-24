"""cache.py against a throwaway SQLite file — exact-match on normalised question text."""

from src import cache
from src.llm import LLMResponse
from src.rag import Citation, RAGResult


def _result(question="What must banks do?") -> RAGResult:
    return RAGResult(
        question=question,
        answer="Banks must comply. [sarb_d3_2023, p.3]",
        citations=[Citation("sarb_d3_2023", 3, True)],
        retrieved_chunks=[],
        refused=False,
        flagged_injection=False,
        llm_response=LLMResponse(
            text="...", model="claude-haiku-4-5", input_tokens=50, output_tokens=20, cost_usd=0.0005
        ),
    )


def test_miss_on_an_unseen_question(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")

    assert cache.get_cached("A question never asked before") is None


def test_set_then_get_returns_the_cached_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")

    cache.set_cached("What must banks do?", _result())
    cached = cache.get_cached("What must banks do?")

    assert cached is not None
    assert cached.answer == "Banks must comply. [sarb_d3_2023, p.3]"
    assert cached.citations == [Citation("sarb_d3_2023", 3, True)]
    assert cached.retrieved_chunks == []  # a cache hit never repopulates retrieval


def test_lookup_is_case_and_whitespace_insensitive(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")

    cache.set_cached("What must banks do?", _result())

    assert cache.get_cached("  WHAT MUST banks DO?  ") is not None


def test_different_question_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")

    cache.set_cached("What must banks do?", _result())

    assert cache.get_cached("What must banks do about hybrid capital instruments?") is None
