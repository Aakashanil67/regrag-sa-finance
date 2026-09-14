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
import os
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    COLLECTION_NAME,
    CROSS_ENCODER_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_TOKENIZER_NAME,
    MANIFEST_PATH,
    RERANK_CANDIDATE_POOL_SIZE,
    RETRIEVAL_STRATEGY,
    ROOT,
    TOKENIZER_AWARE_CHUNK_OVERLAP_TOKENS,
    TOKENIZER_AWARE_CHUNK_TARGET_TOKENS,
    TOKENIZER_ENCODING,
)


class StoreProvenanceError(RuntimeError):
    """Raised when the vector store's recorded build provenance doesn't match the current
    manifest/retrieval config, or no build record exists at all."""


_INDEX_BEHAVIOR_FILES = (
    "src/ingest.py",
    "src/chunking.py",
    "src/store.py",
    "src/config.py",
)
_PIPELINE_BEHAVIOR_FILES = (
    "src/retrieve.py",
    "src/rag.py",
    "src/llm.py",
    "src/config.py",
)
_REQUIREMENTS_FILES = ("requirements-api.txt", "requirements-dev.txt")


def manifest_digest() -> str:
    return hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()


def _file_sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def _runtime_versions() -> dict[str, str]:
    packages = ("chromadb", "sentence-transformers", "pymupdf", "tiktoken")
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _model_revision(env_name: str, model_name: str) -> str:
    return os.environ.get(env_name, f"unresolved:{model_name}")


def _canonical_fingerprint(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def index_fingerprint() -> str:
    """Identify every input that changes extracted chunks or their embeddings."""
    payload = {
        "schema": 2,
        "behavior_files": {path: _file_sha256(path) for path in _INDEX_BEHAVIOR_FILES},
        "requirements": {path: _file_sha256(path) for path in _REQUIREMENTS_FILES},
        "manifest_sha256": manifest_digest(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_model_revision": _model_revision(
            "EMBEDDING_MODEL_REVISION", EMBEDDING_MODEL_NAME
        ),
        "chunk_target": CHUNK_TARGET_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP_TOKENS,
        "tokenizer_encoding": TOKENIZER_ENCODING,
        "embedding_tokenizer": EMBEDDING_TOKENIZER_NAME,
        "tokenizer_aware_chunk_target": TOKENIZER_AWARE_CHUNK_TARGET_TOKENS,
        "tokenizer_aware_chunk_overlap": TOKENIZER_AWARE_CHUNK_OVERLAP_TOKENS,
        "embedding_tokenizer_revision": _model_revision(
            "EMBEDDING_TOKENIZER_REVISION", EMBEDDING_TOKENIZER_NAME
        ),
        "runtime_versions": _runtime_versions(),
    }
    return _canonical_fingerprint(payload)


def _retrieval_config(*, index_identity: str | None = None) -> dict[str, object]:
    return {
        "manifest_sha256": manifest_digest(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": CROSS_ENCODER_MODEL_NAME,
        "collection": COLLECTION_NAME,
        "chunk_target": CHUNK_TARGET_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP_TOKENS,
        "tokenizer_encoding": TOKENIZER_ENCODING,
        "embedding_tokenizer": EMBEDDING_TOKENIZER_NAME,
        "tokenizer_aware_chunk_target": TOKENIZER_AWARE_CHUNK_TARGET_TOKENS,
        "tokenizer_aware_chunk_overlap": TOKENIZER_AWARE_CHUNK_OVERLAP_TOKENS,
        "embedding_tokenizer_revision": _model_revision(
            "EMBEDDING_TOKENIZER_REVISION", EMBEDDING_TOKENIZER_NAME
        ),
        "candidate_pool": RERANK_CANDIDATE_POOL_SIZE,
        "retrieval_strategy": RETRIEVAL_STRATEGY,
        "index_fingerprint": index_identity or index_fingerprint(),
    }


def pipeline_fingerprint(*, k: int) -> str:
    from src.llm import effective_llm_settings
    from src.rag import _SYSTEM_PROMPT, CITATION_CONTRACT_VERSION

    settings = effective_llm_settings()
    payload = {
        "schema": 2,
        "behavior_files": {path: _file_sha256(path) for path in _PIPELINE_BEHAVIOR_FILES},
        "requirements": {path: _file_sha256(path) for path in _REQUIREMENTS_FILES},
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
    return _canonical_fingerprint(payload)


def store_build_record(*, chunk_count: int, index_identity: str | None = None) -> dict[str, object]:
    return {
        "schema": 2,
        "built_at": datetime.now(UTC).isoformat(),
        "chunk_count": chunk_count,
        **_retrieval_config(index_identity=index_identity),
    }


def assert_store_compatible() -> None:
    build_path = CHROMA_DIR / "build.json"
    if not build_path.exists():
        raise StoreProvenanceError(f"{build_path} is missing — run `python -m src.store --rebuild`")

    try:
        record = json.loads(build_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise StoreProvenanceError(f"{build_path} is malformed: {e}") from e

    if record.get("schema") != 2:
        raise StoreProvenanceError(
            f"{build_path} uses schema {record.get('schema')!r}; rebuild the store"
        )

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
