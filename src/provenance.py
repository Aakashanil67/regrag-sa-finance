"""Canonical fingerprints tying a cached, logged, or reported result to the exact pipeline that
produced it.

`pipeline_fingerprint` exists because cache.py's own docstring records a real incident: adopting
chunk_size=800 + reranking rebuilt the vector store and changed retrieval behaviour, but every
previously cached answer kept its old key and kept being served, so the running product silently
disagreed with what the eval reports described. A fingerprint has to include everything that
changes what an answer would be — provider, model, temperature, k, corpus bytes, prompt bytes,
chunking/reranking config, and the citation-contract version — or a change to any of those aliases
under one cache/store identity instead of correctly missing.

`store_build_record`/`assert_store_compatible` apply the same idea to the vector store itself:
`chroma/build.json` records what the store was built from, so readiness and release tooling can
detect "the manifest or retrieval config changed since the last rebuild" without loading either
model.
"""

import hashlib
import json
from datetime import UTC, datetime

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    COLLECTION_NAME,
    CROSS_ENCODER_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    MANIFEST_PATH,
    RERANK_CANDIDATE_POOL_SIZE,
)


class StoreProvenanceError(RuntimeError):
    """Raised when the vector store's recorded build provenance doesn't match the current
    manifest/retrieval config, or no build record exists at all."""


def manifest_digest() -> str:
    return hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()


def _retrieval_config() -> dict[str, object]:
    return {
        "manifest_sha256": manifest_digest(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": CROSS_ENCODER_MODEL_NAME,
        "collection": COLLECTION_NAME,
        "chunk_target": CHUNK_TARGET_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP_TOKENS,
        "candidate_pool": RERANK_CANDIDATE_POOL_SIZE,
    }


def pipeline_fingerprint(*, k: int) -> str:
    from src.llm import effective_llm_settings
    from src.rag import _SYSTEM_PROMPT, CITATION_CONTRACT_VERSION

    settings = effective_llm_settings()
    payload = {
        "schema": 1,
        "citation_contract": CITATION_CONTRACT_VERSION,
        "provider": settings.provider,
        "model": settings.model,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "ollama_host": settings.host,
        "rerank": True,
        "k": k,
        "prompt_sha256": hashlib.sha256(_SYSTEM_PROMPT.encode()).hexdigest(),
        **_retrieval_config(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def store_build_record(*, chunk_count: int) -> dict[str, object]:
    return {
        "schema": 1,
        "built_at": datetime.now(UTC).isoformat(),
        "chunk_count": chunk_count,
        **_retrieval_config(),
    }


def assert_store_compatible() -> None:
    build_path = CHROMA_DIR / "build.json"
    if not build_path.exists():
        raise StoreProvenanceError(f"{build_path} is missing — run `python -m src.store --rebuild`")

    try:
        record = json.loads(build_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise StoreProvenanceError(f"{build_path} is malformed: {e}") from e

    expected = _retrieval_config()
    for key, value in expected.items():
        if record.get(key) != value:
            raise StoreProvenanceError(
                f"store provenance mismatch on {key!r} (built with {record.get(key)!r}, "
                f"current config is {value!r}) — rebuild the store"
            )

    from src.store import get_collection

    actual_count = get_collection().count()
    if record.get("chunk_count") != actual_count:
        raise StoreProvenanceError(
            f"collection has {actual_count} chunks but build.json records "
            f"{record.get('chunk_count')} — rebuild the store"
        )
