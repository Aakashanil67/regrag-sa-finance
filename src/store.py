"""Chunks -> a persistent ChromaDB collection, embedded with sentence-transformers.

An embedding is a fixed-length vector such that texts with similar meaning end up close together
in that vector space — "the bank must hold capital against expected losses" and "provisioning
requirements for anticipated defaults" land near each other even though they share almost no
words. That's what lets retrieval work on meaning instead of exact keyword overlap.
`all-MiniLM-L6-v2` is a small (~80MB) model that runs fast on CPU; it trades some accuracy against
a larger model for a corpus this size (~1,000 chunks) where that accuracy gap doesn't show up in
the retrieval benchmark (see reports/retrieval_bench.md).

Rebuilding is idempotent and incremental: each chunk's own content hash (`Chunk.chunk_hash`, from
chunking.py) is its Chroma document ID, so re-running after editing one document only embeds the
chunks that actually changed — everything else is already sitting in the collection under the same
ID and gets left alone. Chunks whose hash disappeared (the source document changed or dropped
them) are deleted, so the store never silently accumulates stale vectors from a previous version
of a document.

    python -m src.store              # report current collection stats, no changes
    python -m src.store --rebuild    # ingest anything new/changed, drop anything stale
"""

import argparse
import json
import os
import tempfile
import time

from src.chunking import Chunk, chunk_corpus
from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

_model = None


def _get_model():
    # deferred: torch + sentence-transformers are a heavy, slow-to-install dependency that most
    # tests (and any code just reading Chunk/RetrievedChunk types) never need to pay for
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_collection():
    import chromadb
    from chromadb.config import Settings

    # No reason for a local research tool to phone home. Doesn't actually suppress the
    # "Failed to send telemetry event" stderr noise chromadb's telemetry code produces against
    # this project's pinned posthog version — see requirements.txt's posthog comment — but it's
    # still the right default regardless of whether that call would otherwise succeed.
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(COLLECTION_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """The one place text turns into a vector — retrieve.py embeds queries through this same
    function so a query and the chunks it's compared against always share a model."""
    return _get_model().encode(texts, show_progress_bar=False).tolist()


def _chunk_metadata(chunk: Chunk) -> dict:
    return {
        "doc_id": chunk.doc_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section": chunk.section or "",
        "token_count": chunk.token_count,
    }


def rebuild() -> dict:
    all_chunks = [c for chunks in chunk_corpus().values() for c in chunks]
    desired = {c.chunk_hash: c for c in all_chunks}

    collection = get_collection()
    existing_ids = set(collection.get(include=[])["ids"]) if collection.count() else set()

    to_delete = existing_ids - desired.keys()
    to_add_ids = list(desired.keys() - existing_ids)

    if to_delete:
        collection.delete(ids=list(to_delete))

    embed_seconds = 0.0
    if to_add_ids:
        chunks_to_add = [desired[chunk_id] for chunk_id in to_add_ids]
        start = time.perf_counter()
        embeddings = embed_texts([c.text for c in chunks_to_add])
        embed_seconds = time.perf_counter() - start

        collection.add(
            ids=to_add_ids,
            embeddings=embeddings,
            documents=[c.text for c in chunks_to_add],
            metadatas=[_chunk_metadata(c) for c in chunks_to_add],
        )

    stats = {
        "added": len(to_add_ids),
        "deleted": len(to_delete),
        "unchanged": len(desired) - len(to_add_ids),
        "total": collection.count(),
        "embed_seconds": embed_seconds,
    }
    _write_build_record(stats["total"])
    return stats


def _write_build_record(chunk_count: int) -> None:
    """Written last, after ingestion has already succeeded, and via a temp-file-then-replace swap
    so an interrupted rebuild can never leave a build.json that claims a build finished when it
    didn't — see src/provenance.py, which readiness/release tooling trusts this file to reflect."""
    from src.provenance import store_build_record

    record = store_build_record(chunk_count=chunk_count)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=CHROMA_DIR, prefix=".build_", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(record, f, indent=2, sort_keys=True)
        os.replace(tmp_path, CHROMA_DIR / "build.json")
    except BaseException:
        os.unlink(tmp_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="ingest new/changed chunks")
    parser.add_argument(
        "--check", action="store_true", help="verify store provenance matches current config"
    )
    args = parser.parse_args()

    if args.rebuild:
        stats = rebuild()
        print(
            f"added {stats['added']}, deleted {stats['deleted']}, "
            f"unchanged {stats['unchanged']}, total {stats['total']} chunks "
            f"(embedding took {stats['embed_seconds']:.1f}s)"
        )
    elif args.check:
        from src.provenance import (
            StoreProvenanceError,
            assert_store_compatible,
            pipeline_fingerprint,
        )

        try:
            assert_store_compatible()
        except StoreProvenanceError as e:
            print(f"store provenance check failed: {e}")
            raise SystemExit(1) from e
        collection = get_collection()
        print(f"collection '{COLLECTION_NAME}': {collection.count()} chunks stored")
        print(f"fingerprint: {pipeline_fingerprint(k=5)[:16]}...")
    else:
        collection = get_collection()
        print(f"collection '{COLLECTION_NAME}': {collection.count()} chunks stored")
        print(f"embedding model: {EMBEDDING_MODEL_NAME}")


if __name__ == "__main__":
    main()
