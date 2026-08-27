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
    cached = cache.get_cached("  WHAT MUST banks DO?  ")

    assert cached is not None
    assert cached.answer == "Banks must comply. [sarb_d3_2023, p.3]"


def test_different_question_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")

    cache.set_cached("What must banks do?", _result())

    assert cache.get_cached("What must banks do about hybrid capital instruments?") is None


def test_a_retrieval_config_change_invalidates_previously_cached_answers(tmp_path, monkeypatch):
    # the real scenario: adopting chunk_size=800 + reranking rebuilt the store and changed what
    # retrieval returns, but a question-text-only key kept serving the answers generated under the
    # old config. Every cached answer stayed "valid" forever while the eval reports described
    # different behaviour — invisible unless you diff a live answer against a fresh one.
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")
    cache.set_cached("What must banks do?", _result())
    assert cache.get_cached("What must banks do?") is not None

    monkeypatch.setattr(cache, "CHUNK_TARGET_TOKENS", 500)

    assert cache.get_cached("What must banks do?") is None


def test_cache_hit_still_carries_source_notices(tmp_path, monkeypatch):
    # get_cached used to hardcode flagged_injection=False; source_notices is the same shape of
    # bug waiting to happen if it were ever stored as a stale snapshot instead of recomputed —
    # it's derived from citations_json, which the cache does store, so it must survive a hit
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")
    result = RAGResult(
        question="What impairment model does IFRS 9 use?",
        answer="An expected-loss model. [pwc_practical_guide_ifrs9, p.1]",
        citations=[Citation("pwc_practical_guide_ifrs9", 1, True)],
        retrieved_chunks=[],
        refused=False,
        flagged_injection=False,
        llm_response=LLMResponse(
            text="...", model="claude-haiku-4-5", input_tokens=50, output_tokens=20, cost_usd=0.0005
        ),
    )
    cache.set_cached("What impairment model does IFRS 9 use?", result)

    cached = cache.get_cached("What impairment model does IFRS 9 use?")

    assert len(cached.source_notices) == 1
    assert "third-party commentary" in cached.source_notices[0]


def test_a_system_prompt_edit_invalidates_previously_cached_answers(tmp_path, monkeypatch):
    # same failure mode via the other input: the prompt carries the citation and refusal rules, so
    # editing it changes what an answer looks like just as surely as re-chunking does
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")
    cache.set_cached("What must banks do?", _result())

    monkeypatch.setattr(cache, "_SYSTEM_PROMPT", "A materially different system prompt.")

    assert cache.get_cached("What must banks do?") is None
