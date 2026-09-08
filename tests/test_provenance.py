"""Every input that changes what an answer would be must change pipeline_fingerprint(), or a
cache/store/report keyed on it silently serves output from a different configuration — see
src/cache.py's docstring for the incident that made this the rule. These tests prove each
individual input is actually wired in, one at a time, rather than trusting the payload shape.
"""

import json

from src import provenance


def test_provider_and_effective_model_change_pipeline_fingerprint(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    openai_fp = provenance.pipeline_fingerprint(k=5)

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    ollama_fp = provenance.pipeline_fingerprint(k=5)

    assert openai_fp != ollama_fp


def test_retrieval_k_changes_pipeline_fingerprint():
    assert provenance.pipeline_fingerprint(k=5) != provenance.pipeline_fingerprint(k=10)


def test_temperature_changes_pipeline_fingerprint(monkeypatch):
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    default_fp = provenance.pipeline_fingerprint(k=5)

    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    hot_fp = provenance.pipeline_fingerprint(k=5)

    assert default_fp != hot_fp


def test_chunk_overlap_changes_pipeline_fingerprint(monkeypatch):
    base_fp = provenance.pipeline_fingerprint(k=5)

    monkeypatch.setattr(provenance, "CHUNK_OVERLAP_TOKENS", 999)

    assert provenance.pipeline_fingerprint(k=5) != base_fp


def test_rerank_candidate_pool_changes_pipeline_fingerprint(monkeypatch):
    base_fp = provenance.pipeline_fingerprint(k=5)

    monkeypatch.setattr(provenance, "RERANK_CANDIDATE_POOL_SIZE", 999)

    assert provenance.pipeline_fingerprint(k=5) != base_fp


def test_manifest_bytes_change_pipeline_fingerprint(monkeypatch, tmp_path):
    original = provenance.MANIFEST_PATH.read_bytes()
    fake_manifest = tmp_path / "manifest.json"
    fake_manifest.write_bytes(original)

    monkeypatch.setattr(provenance, "MANIFEST_PATH", fake_manifest)
    base_fp = provenance.pipeline_fingerprint(k=5)

    fake_manifest.write_bytes(original + b" ")
    assert provenance.pipeline_fingerprint(k=5) != base_fp


def test_prompt_bytes_change_pipeline_fingerprint(monkeypatch):
    from src import rag

    base_fp = provenance.pipeline_fingerprint(k=5)
    monkeypatch.setattr(rag, "_SYSTEM_PROMPT", rag._SYSTEM_PROMPT + " extra rule")

    assert provenance.pipeline_fingerprint(k=5) != base_fp


def test_citation_contract_version_changes_pipeline_fingerprint(monkeypatch):
    from src import rag

    base_fp = provenance.pipeline_fingerprint(k=5)
    monkeypatch.setattr(rag, "CITATION_CONTRACT_VERSION", rag.CITATION_CONTRACT_VERSION + 1)

    assert provenance.pipeline_fingerprint(k=5) != base_fp


def test_manifest_digest_is_a_sha256_hex_digest():
    digest = provenance.manifest_digest()

    assert len(digest) == 64
    int(digest, 16)  # raises ValueError if not valid hex


def test_store_build_record_contains_the_retrieval_relevant_fields():
    record = provenance.store_build_record(chunk_count=123)

    assert record["chunk_count"] == 123
    assert record["manifest_sha256"] == provenance.manifest_digest()
    assert record["collection"] == provenance.COLLECTION_NAME
    assert "built_at" in record


def test_assert_store_compatible_raises_when_build_record_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "CHROMA_DIR", tmp_path)

    import pytest

    with pytest.raises(provenance.StoreProvenanceError):
        provenance.assert_store_compatible()


def test_assert_store_compatible_raises_when_manifest_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "CHROMA_DIR", tmp_path)
    record = provenance.store_build_record(chunk_count=1)
    record["manifest_sha256"] = "0" * 64
    (tmp_path / "build.json").write_text(json.dumps(record), encoding="utf-8")

    import pytest

    with pytest.raises(provenance.StoreProvenanceError):
        provenance.assert_store_compatible()
