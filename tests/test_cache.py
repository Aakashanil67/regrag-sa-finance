"""cache.py against a throwaway SQLite file — exact-match on normalised question text, keyed by a
provenance fingerprint that must change whenever the effective pipeline changes (src/provenance.py
owns that computation; cache.py must not keep a second, drifting copy of it — see src/cache.py's
own docstring for the incident that made this the rule)."""

import sqlite3

import pytest

from src import cache
from src.llm import LLMResponse
from src.rag import Citation, RAGResult, RefusalReason


@pytest.fixture(autouse=True)
def _enable_cache_for_existing_cache_contracts(monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "true")


def _result(question="What must banks do?", refusal_reason=None) -> RAGResult:
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
        refusal_reason=refusal_reason,
    )


def test_miss_on_an_unseen_question(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")

    assert cache.get_cached("A question never asked before", k=5) is None


def test_disabled_cache_never_opens_database(tmp_path, monkeypatch):
    path = tmp_path / "cache.sqlite3"
    monkeypatch.setattr(cache, "CACHE_DB_PATH", path)
    monkeypatch.delenv("CACHE_ENABLED", raising=False)

    cache.set_cached("PRIVATE_MARKER", _result("PRIVATE_MARKER"), k=5)

    assert cache.get_cached("PRIVATE_MARKER", k=5) is None
    assert not path.exists()


def test_set_then_get_returns_the_cached_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")

    cache.set_cached("What must banks do?", _result(), k=5)
    cached = cache.get_cached("What must banks do?", k=5)

    assert cached is not None
    assert cached.answer == "Banks must comply. [sarb_d3_2023, p.3]"
    assert cached.citations == [Citation("sarb_d3_2023", 3, True)]
    assert cached.retrieved_chunks == []  # a cache hit never repopulates retrieval


def test_lookup_is_case_and_whitespace_insensitive(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")

    cache.set_cached("What must banks do?", _result(), k=5)
    cached = cache.get_cached("  WHAT MUST banks DO?  ", k=5)

    assert cached is not None
    assert cached.answer == "Banks must comply. [sarb_d3_2023, p.3]"


def test_different_question_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")

    cache.set_cached("What must banks do?", _result(), k=5)

    assert cache.get_cached("What must banks do about hybrid capital instruments?", k=5) is None


def test_a_different_k_is_a_miss(tmp_path, monkeypatch):
    # k changes what retrieval returns, so it must change what a cached answer means, exactly like
    # a chunk-size or reranker change does — this was previously not part of the key at all
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")
    cache.set_cached("What must banks do?", _result(), k=5)

    assert cache.get_cached("What must banks do?", k=10) is None


def test_a_provider_model_switch_invalidates_previously_cached_answers(tmp_path, monkeypatch):
    # the exact audited sequence that previously produced one shared fingerprint across all three
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    cache.set_cached("What must banks do?", _result(), k=5)
    assert cache.get_cached("What must banks do?", k=5) is not None

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    assert cache.get_cached("What must banks do?", k=5) is None
    cache.set_cached("What must banks do?", _result(), k=5)

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5")
    assert cache.get_cached("What must banks do?", k=5) is None


def test_a_retrieval_config_change_invalidates_previously_cached_answers(tmp_path, monkeypatch):
    # the real scenario: adopting chunk_size=800 + reranking rebuilt the store and changed what
    # retrieval returns, but a question-text-only key kept serving the answers generated under the
    # old config. Every cached answer stayed "valid" forever while the eval reports described
    # different behaviour — invisible unless you diff a live answer against a fresh one.
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")
    cache.set_cached("What must banks do?", _result(), k=5)
    assert cache.get_cached("What must banks do?", k=5) is not None

    from src import provenance

    monkeypatch.setattr(provenance, "CHUNK_TARGET_TOKENS", 500)

    assert cache.get_cached("What must banks do?", k=5) is None


def test_a_temperature_change_invalidates_previously_cached_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")
    cache.set_cached("What must banks do?", _result(), k=5)
    assert cache.get_cached("What must banks do?", k=5) is not None

    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")

    assert cache.get_cached("What must banks do?", k=5) is None


def test_a_citation_contract_version_bump_invalidates_previously_cached_answers(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")
    cache.set_cached("What must banks do?", _result(), k=5)
    assert cache.get_cached("What must banks do?", k=5) is not None

    from src import rag

    monkeypatch.setattr(rag, "CITATION_CONTRACT_VERSION", rag.CITATION_CONTRACT_VERSION + 1)

    assert cache.get_cached("What must banks do?", k=5) is None


def test_cache_hit_still_carries_source_notices(tmp_path, monkeypatch):
    # get_cached used to hardcode flagged_injection=False; source_notices is the same shape of
    # bug waiting to happen if it were ever stored as a stale snapshot instead of recomputed —
    # it's derived from citations_json, which the cache does store, so it must survive a hit
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")
    from src import rag as rag_module

    monkeypatch.setattr(
        rag_module,
        "_doc_metadata",
        lambda: {"mock_third_party_doc": {"is_third_party": True, "document_type": "Commentary"}},
    )
    result = RAGResult(
        question="What impairment model does IFRS 9 use?",
        answer="An expected-loss model. [mock_third_party_doc, p.1]",
        citations=[Citation("mock_third_party_doc", 1, True)],
        retrieved_chunks=[],
        refused=False,
        flagged_injection=False,
        llm_response=LLMResponse(
            text="...", model="claude-haiku-4-5", input_tokens=50, output_tokens=20, cost_usd=0.0005
        ),
    )
    cache.set_cached("What impairment model does IFRS 9 use?", result, k=5)

    cached = cache.get_cached("What impairment model does IFRS 9 use?", k=5)

    assert len(cached.source_notices) == 1
    assert "third-party commentary" in cached.source_notices[0].text


def test_a_system_prompt_edit_invalidates_previously_cached_answers(tmp_path, monkeypatch):
    # same failure mode via the other input: the prompt carries the citation and refusal rules, so
    # editing it changes what an answer looks like just as surely as re-chunking does
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")
    cache.set_cached("What must banks do?", _result(), k=5)

    from src import rag

    monkeypatch.setattr(rag, "_SYSTEM_PROMPT", "A materially different system prompt.")

    assert cache.get_cached("What must banks do?", k=5) is None


def test_refusal_reason_round_trips_through_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")
    result = _result(refusal_reason=RefusalReason.UNVERIFIED_CITATION)
    cache.set_cached("What must banks do?", result, k=5)

    cached = cache.get_cached("What must banks do?", k=5)

    assert cached.refusal_reason == RefusalReason.UNVERIFIED_CITATION


def test_an_accepted_answer_has_no_refusal_reason_in_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.sqlite3")
    cache.set_cached("What must banks do?", _result(), k=5)

    cached = cache.get_cached("What must banks do?", k=5)

    assert cached.refusal_reason is None


def test_v2_cache_has_no_question_column_and_does_not_serve_legacy_rows(tmp_path, monkeypatch):
    path = tmp_path / "cache.sqlite3"
    monkeypatch.setattr(cache, "CACHE_DB_PATH", path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE response_cache (question_hash TEXT PRIMARY KEY, question TEXT, answer TEXT)"
        )
        conn.execute(
            "INSERT INTO response_cache VALUES ('legacy-key', 'PRIVATE_QUESTION', 'legacy answer')"
        )

    assert cache.get_cached("PRIVATE_QUESTION", k=5) is None
    cache.set_cached("Current question", _result(), k=5)

    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(response_cache_v2)")}
        assert "question" not in columns
        assert conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM response_cache_v2").fetchone()[0] == 1


def test_expired_cache_entry_is_not_served_at_the_exact_boundary(tmp_path, monkeypatch):
    path = tmp_path / "cache.sqlite3"
    monkeypatch.setattr(cache, "CACHE_DB_PATH", path)
    monkeypatch.setenv("CACHE_TTL_SECONDS", "10")
    clock = [100.0]
    monkeypatch.setattr(cache.time, "time", lambda: clock[0])

    cache.set_cached("What must banks do?", _result(), k=5)
    clock[0] = 110.0

    assert cache.get_cached("What must banks do?", k=5) is None


def test_refusal_result_round_trips_with_expiry(tmp_path, monkeypatch):
    path = tmp_path / "cache.sqlite3"
    monkeypatch.setattr(cache, "CACHE_DB_PATH", path)
    refusal = RAGResult(
        question="Unknown",
        answer="I don't have a source for that.",
        citations=[],
        retrieved_chunks=[],
        refused=True,
        flagged_injection=False,
        llm_response=LLMResponse(
            text="INSUFFICIENT", model="fake", input_tokens=1, output_tokens=1, cost_usd=0.0
        ),
        refusal_reason=RefusalReason.NO_CONTEXT,
    )

    cache.set_cached("Unknown", refusal, k=5)
    cached = cache.get_cached("Unknown", k=5)

    assert cached is not None
    assert cached.refused is True
    assert cached.refusal_reason is RefusalReason.NO_CONTEXT


def test_cache_purge_expired_supports_dry_run_and_removes_only_expired_rows(tmp_path, monkeypatch):
    path = tmp_path / "cache.sqlite3"
    monkeypatch.setattr(cache, "CACHE_DB_PATH", path)
    monkeypatch.setenv("CACHE_TTL_SECONDS", "10")
    clock = [100.0]
    monkeypatch.setattr(cache.time, "time", lambda: clock[0])
    cache.set_cached("old", _result("old"), k=5)
    clock[0] = 105.0
    cache.set_cached("new", _result("new"), k=5)
    clock[0] = 110.0

    assert cache.purge_expired(now=110.0, dry_run=True) == 1
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM response_cache_v2").fetchone()[0] == 2
    assert cache.purge_expired(now=110.0) == 1
    assert cache.get_cached("old", k=5) is None
    assert cache.get_cached("new", k=5) is not None
    assert cache.purge_expired(now=110.0) == 0


def test_scrub_cache_removes_legacy_and_v2_rows_even_when_disabled(tmp_path, monkeypatch):
    path = tmp_path / "cache.sqlite3"
    monkeypatch.setattr(cache, "CACHE_DB_PATH", path)
    cache.set_cached("new", _result("new"), k=5)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE response_cache (question_hash TEXT PRIMARY KEY, question TEXT, answer TEXT)"
        )
        conn.execute("INSERT INTO response_cache VALUES ('old', 'old question', 'old answer')")
    monkeypatch.delenv("CACHE_ENABLED", raising=False)

    assert cache.scrub_cache(dry_run=True) == {"response_cache": 1, "response_cache_v2": 1}
    assert cache.scrub_cache() == {"response_cache": 1, "response_cache_v2": 1}
    assert cache.scrub_cache() == {"response_cache": 0, "response_cache_v2": 0}
