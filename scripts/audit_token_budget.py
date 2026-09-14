"""Measure model-token truncation in the existing development store.

This command is deliberately read-only with respect to the corpus and Chroma store. It loads the
actual local embedding and reranker tokenizers, measures the text that the models would receive,
and writes a JSON audit. It never calls an answer model and refuses every split except ``dev``.

    python -m scripts.audit_token_budget --split dev --output reports/token_budget_baseline.json
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from src import config
from src.chunking import get_embedding_tokenizer
from src.retrieve import _fetch_candidates, _get_cross_encoder
from src.store import _get_model, _load_build_record, get_collection


def validate_split(split: str) -> None:
    if split != "dev":
        raise ValueError("token-budget audit accepts the development split only")


def percentiles(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}

    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    return {
        "count": len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": max(ordered),
    }


def _resolved_identity(obj, requested: str) -> str:
    for candidate in (
        getattr(obj, "_commit_hash", None),
        getattr(obj, "revision", None),
        getattr(obj, "init_kwargs", {}).get("_commit_hash")
        if hasattr(obj, "init_kwargs")
        else None,
        os.environ.get("EMBEDDING_MODEL_REVISION"),
    ):
        if candidate:
            return str(candidate)
    return f"unresolved:{requested}"


def _max_length(model, tokenizer) -> int | None:
    values = []
    getter = getattr(model, "get_max_seq_length", None)
    if getter is not None:
        values.append(getter())
    values.extend(
        [
            getattr(model, "max_length", None),
            getattr(tokenizer, "model_max_length", None),
        ]
    )
    for value in values:
        if isinstance(value, int) and value > 0 and value < 1_000_000:
            return value
    return None


def _encode_length(tokenizer, text: str, *, pair: str | None = None) -> int:
    if pair is None:
        return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    return len(
        tokenizer.encode(
            text,
            text_pair=pair,
            add_special_tokens=True,
            truncation=False,
        )
    )


def _embedding_audit(collection, tokenizer, model, questions: list[dict]) -> dict:
    rows = collection.get(include=["documents", "metadatas"])
    limit = _max_length(model, tokenizer)
    all_lengths = []
    for document, _metadata in zip(rows["documents"], rows["metadatas"], strict=True):
        length = _encode_length(tokenizer, document)
        all_lengths.append(length)

    query_lengths = [_encode_length(tokenizer, item["question"]) for item in questions]
    truncated_chunks = sum(limit is not None and length > limit for length in all_lengths)
    truncated_queries = sum(limit is not None and length > limit for length in query_lengths)

    # Keep per-document truncation counts without duplicating the public summary shape.
    per_doc = {}
    for document, metadata in zip(rows["documents"], rows["metadatas"], strict=True):
        doc_id = metadata["doc_id"]
        stats = per_doc.setdefault(doc_id, [])
        stats.append(_encode_length(tokenizer, document))
    per_doc_summary = {
        doc_id: {
            "chunk_count": len(lengths),
            "lengths": percentiles(lengths),
            "truncated_chunks": sum(limit is not None and length > limit for length in lengths),
        }
        for doc_id, lengths in sorted(per_doc.items())
    }
    return {
        "model": config.EMBEDDING_MODEL_NAME,
        "tokenizer": config.EMBEDDING_TOKENIZER_NAME,
        "tokenizer_revision": _resolved_identity(tokenizer, config.EMBEDDING_TOKENIZER_NAME),
        "max_sequence_length": limit,
        "chunk_lengths": percentiles(all_lengths),
        "truncated_chunks": truncated_chunks,
        "truncated_chunk_percent": (truncated_chunks / len(all_lengths) if all_lengths else 0.0),
        "query_lengths": percentiles(query_lengths),
        "truncated_queries": truncated_queries,
        "truncated_query_percent": (
            truncated_queries / len(query_lengths) if query_lengths else 0.0
        ),
        "per_document": per_doc_summary,
    }


def _reranker_audit(collection, tokenizer, model, questions: list[dict]) -> dict:
    limit = _max_length(model, tokenizer)
    lengths = []
    sample_pairs = []
    for item in questions:
        candidates = _fetch_candidates(
            item["question"], config.RERANK_CANDIDATE_POOL_SIZE, None, collection=collection
        )
        for candidate in candidates:
            if len(sample_pairs) >= 100:
                break
            lengths.append(_encode_length(tokenizer, item["question"], pair=candidate.text))
            sample_pairs.append({"question_id": item["id"], "doc_id": candidate.doc_id})
        if len(sample_pairs) >= 100:
            break
    truncated = sum(limit is not None and length > limit for length in lengths)
    return {
        "model": config.CROSS_ENCODER_MODEL_NAME,
        "tokenizer": getattr(tokenizer, "name_or_path", config.CROSS_ENCODER_MODEL_NAME),
        "tokenizer_revision": _resolved_identity(tokenizer, config.CROSS_ENCODER_MODEL_NAME),
        "max_sequence_length": limit,
        "sample_size": len(lengths),
        "pair_lengths": percentiles(lengths),
        "truncated_pairs": truncated,
        "truncated_pair_percent": (truncated / len(lengths) if lengths else 0.0),
        "sample_pairs": sample_pairs,
    }


def audit(split: str) -> dict:
    validate_split(split)
    questions = json.loads(config.RETRIEVAL_DEV_PATH.read_text(encoding="utf-8"))
    collection = get_collection()
    embedding_tokenizer = get_embedding_tokenizer()
    embedding_model = _get_model()
    cross_encoder = _get_cross_encoder()
    result = {
        "schema": 1,
        "audited_at": datetime.now(UTC).isoformat(),
        "split": split,
        "read_only": True,
        "answer_model_called": False,
        "store": {
            "collection_name": collection.name,
            "chunk_count": collection.count(),
            "build_record": _load_build_record(),
        },
        "embedding": _embedding_audit(collection, embedding_tokenizer, embedding_model, questions),
        "reranker": _reranker_audit(collection, cross_encoder.tokenizer, cross_encoder, questions),
        "budget_configuration": {
            "tokenizer_aware_target": config.TOKENIZER_AWARE_CHUNK_TARGET_TOKENS,
            "tokenizer_aware_overlap": config.TOKENIZER_AWARE_CHUNK_OVERLAP_TOKENS,
        },
    }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit(args.split)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(
        "embedding truncated chunks: "
        f"{result['embedding']['truncated_chunks']}/{result['embedding']['chunk_lengths']['count']}"
    )
    print(
        "reranker truncated pairs: "
        f"{result['reranker']['truncated_pairs']}/{result['reranker']['sample_size']}"
    )


if __name__ == "__main__":
    main()
