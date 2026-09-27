"""Canonical fingerprints tying a cached, logged or reported result to the pipeline that made it.

`pipeline_fingerprint` covers everything that changes an answer: provider, model, temperature, k,
corpus bytes, prompt bytes, chunking and reranking config, and the citation-contract version. If any
of these is left out, a change aliases under one cache identity instead of missing.

`store_build_record` and `assert_store_compatible` do the same for the vector store:
`chroma/build.json` records what it was built from, so a stale store is detectable without loading
a model.
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
    CROSS_ENCODER_MODEL_REVISION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_TOKENIZER_NAME,
    EMBEDDING_TOKENIZER_REVISION,
    MANIFEST_PATH,
    RERANK_CANDIDATE_POOL_SIZE,
    RETRIEVAL_STRATEGY,
    ROOT,
    TOKENIZER_AWARE_CHUNK_OVERLAP_TOKENS,
    TOKENIZER_AWARE_CHUNK_TARGET_TOKENS,
    TOKENIZER_ENCODING,
)

RERANKER_MODEL_REVISION = CROSS_ENCODER_MODEL_REVISION


class StoreProvenanceError(RuntimeError):
    """Raised when the vector store's recorded build provenance doesn't match the current
    manifest/retrieval config, or no build record exists at all."""


class ModelProvenanceError(RuntimeError):
    """Raised when a loaded local model does not expose the configured snapshot revision."""


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


def _model_revision(env_name: str, default_revision: str) -> str:
    return os.environ.get(env_name, default_revision)


def pinned_model_revisions() -> dict[str, str]:
    """Return the configured revisions used by local model loaders and fingerprints."""
    return {
        "embedding": _model_revision("EMBEDDING_MODEL_REVISION", EMBEDDING_MODEL_REVISION),
        "embedding_tokenizer": _model_revision(
            "EMBEDDING_TOKENIZER_REVISION", EMBEDDING_TOKENIZER_REVISION
        ),
        "reranker": _model_revision("RERANKER_MODEL_REVISION", CROSS_ENCODER_MODEL_REVISION),
    }


def _loaded_revision(model, role: str) -> str | None:
    if role == "embedding":
        for module in getattr(model, "_modules", {}).values():
            config = getattr(getattr(module, "auto_model", None), "config", None)
            revision = getattr(config, "_commit_hash", None)
            if revision:
                return revision
    elif role == "reranker":
        config = getattr(getattr(model, "model", None), "config", None)
        revision = getattr(config, "_commit_hash", None)
        if revision:
            return revision
    else:
        raise ValueError(f"unknown local model role: {role!r}")
    return None


def assert_loaded_model_revision(model, role: str) -> dict[str, str]:
    """Verify a loaded model exposes the exact configured snapshot revision."""
    expected = pinned_model_revisions()[role]
    actual = _loaded_revision(model, role)
    if not actual:
        raise ModelProvenanceError(f"{role} loaded model revision is unresolved")
    if actual != expected:
        raise ModelProvenanceError(
            f"{role} loaded model revision {actual!r} does not match pinned {expected!r}"
        )
    names = {"embedding": EMBEDDING_MODEL_NAME, "reranker": CROSS_ENCODER_MODEL_NAME}
    return {"model": names[role], "revision": actual}


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
        "embedding_model_revision": pinned_model_revisions()["embedding"],
        "chunk_target": CHUNK_TARGET_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP_TOKENS,
        "tokenizer_encoding": TOKENIZER_ENCODING,
        "embedding_tokenizer": EMBEDDING_TOKENIZER_NAME,
        "tokenizer_aware_chunk_target": TOKENIZER_AWARE_CHUNK_TARGET_TOKENS,
        "tokenizer_aware_chunk_overlap": TOKENIZER_AWARE_CHUNK_OVERLAP_TOKENS,
        "embedding_tokenizer_revision": pinned_model_revisions()["embedding_tokenizer"],
        "runtime_versions": _runtime_versions(),
    }
    return _canonical_fingerprint(payload)


def _retrieval_config(*, index_identity: str | None = None) -> dict[str, object]:
    return {
        "manifest_sha256": manifest_digest(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_model_revision": pinned_model_revisions()["embedding"],
        "reranker_model": CROSS_ENCODER_MODEL_NAME,
        "reranker_model_revision": pinned_model_revisions()["reranker"],
        "collection": COLLECTION_NAME,
        "chunk_target": CHUNK_TARGET_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP_TOKENS,
        "tokenizer_encoding": TOKENIZER_ENCODING,
        "embedding_tokenizer": EMBEDDING_TOKENIZER_NAME,
        "tokenizer_aware_chunk_target": TOKENIZER_AWARE_CHUNK_TARGET_TOKENS,
        "tokenizer_aware_chunk_overlap": TOKENIZER_AWARE_CHUNK_OVERLAP_TOKENS,
        "embedding_tokenizer_revision": pinned_model_revisions()["embedding_tokenizer"],
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
        "model_revisions": pinned_model_revisions(),
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
