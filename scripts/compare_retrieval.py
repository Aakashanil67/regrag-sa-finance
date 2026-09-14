"""Compare the three predeclared development retrieval variants in isolated stores.

The comparison is intentionally finite and retrieval-only. It never reads a holdout file, calls
an answer/judge model, or changes the serving store/configuration. All variants use the same
corpus, local model instances, candidate budget, question order, and final k.

    python -m scripts.compare_retrieval --split dev --output reports/retrieval_comparison.json
"""

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from evals.retrieval_bench import source_coverage
from scripts.sweep_chunk_size import _build_isolated_collection
from src import config
from src.chunking import get_embedding_tokenizer
from src.provenance import _file_sha256, _model_revision, manifest_digest
from src.retrieve import _get_cross_encoder, retrieve
from src.store import _get_model

K = 5
VARIANTS = (
    {
        "label": "A",
        "chunk_target_tokens": 800,
        "chunk_overlap_tokens": 75,
        "tokenizer_mode": "tiktoken",
        "strategy": "semantic",
        "simplicity": 0,
    },
    {
        "label": "B",
        "chunk_target_tokens": 240,
        "chunk_overlap_tokens": 32,
        "tokenizer_mode": "embedding_wordpiece",
        "strategy": "semantic",
        "simplicity": 0,
    },
    {
        "label": "C",
        "chunk_target_tokens": 240,
        "chunk_overlap_tokens": 32,
        "tokenizer_mode": "embedding_wordpiece",
        "strategy": "named_balanced",
        "simplicity": 1,
    },
)


def validate_split(split: str) -> None:
    if split != "dev":
        raise ValueError("retrieval comparison accepts the development split only")


def _percentiles(values: list[float]) -> dict[str, float | int | None]:
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


def _max_length(model, tokenizer) -> int | None:
    getter = getattr(model, "get_max_seq_length", None)
    candidates = [getter() if getter else None, getattr(model, "max_length", None)]
    candidates.append(getattr(tokenizer, "model_max_length", None))
    return next(
        (value for value in candidates if isinstance(value, int) and 0 < value < 1_000_000),
        None,
    )


def _identity_payload(variant: dict) -> tuple[dict, dict]:
    index_files = ("src/ingest.py", "src/chunking.py", "src/store.py", "src/config.py")
    pipeline_files = ("src/retrieve.py", "src/rag.py", "src/config.py")
    common = {
        "schema": 1,
        "manifest_sha256": manifest_digest(),
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embedding_model_revision": _model_revision(
            "EMBEDDING_MODEL_REVISION", config.EMBEDDING_MODEL_NAME
        ),
        "reranker_model": config.CROSS_ENCODER_MODEL_NAME,
        "reranker_model_revision": _model_revision(
            "RERANKER_MODEL_REVISION", config.CROSS_ENCODER_MODEL_NAME
        ),
        "variant": variant,
        "candidate_pool": config.RERANK_CANDIDATE_POOL_SIZE,
        "k": K,
        "rerank": True,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }
    index_payload = {
        **common,
        "behavior_files": {path: _file_sha256(path) for path in index_files},
    }
    pipeline_payload = {
        **common,
        "behavior_files": {path: _file_sha256(path) for path in pipeline_files},
        "index_identity": hashlib.sha256(
            json.dumps(index_payload, sort_keys=True).encode()
        ).hexdigest(),
    }
    return index_payload, pipeline_payload


def _chunk_record(chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "score": chunk.score,
    }


def _evaluate_item(
    item: dict,
    *,
    item_set: str,
    collection,
    strategy: str,
    embedding_tokenizer,
    embedding_limit,
    reranker_tokenizer,
    reranker_limit,
) -> dict:
    required_sources = (
        item["source"] if "source" in item else [{"doc_id": item["doc_id"], "page": item["page"]}]
    )
    results = retrieve(
        item["question"],
        k=K,
        rerank=True,
        strategy=strategy,
        collection=collection,
    )
    coverage = source_coverage(results, required_sources)
    embedding_lengths = [_encode_length(embedding_tokenizer, chunk.text) for chunk in results]
    pair_lengths = [
        _encode_length(reranker_tokenizer, item["question"], pair=chunk.text) for chunk in results
    ]
    return {
        "set": item_set,
        "id": item["id"],
        "question": item["question"],
        "required_sources": required_sources,
        **coverage,
        "retrieved": [_chunk_record(chunk) for chunk in results],
        "retrieved_count": len(results),
        "embedding_retrieved_lengths": embedding_lengths,
        "reranker_pair_lengths": pair_lengths,
        "embedding_truncated_retrieved": sum(
            embedding_limit is not None and length > embedding_limit for length in embedding_lengths
        ),
        "reranker_truncated_retrieved": sum(
            reranker_limit is not None and length > reranker_limit for length in pair_lengths
        ),
    }


def _summarise(items: list[dict]) -> dict:
    answerable = [item for item in items if item["required_sources"]]
    multi_doc = [item for item in answerable if len(item["required_sources"]) > 1]
    return {
        "count": len(items),
        "answerable_count": len(answerable),
        "any_count": sum(item["any_required_source_hit"] for item in answerable),
        "all_count": sum(item["all_required_source_hit"] for item in answerable),
        "mrr_first_relevant": sum(item["first_relevant_mrr"] for item in answerable)
        / len(answerable)
        if answerable
        else 0.0,
        "multi_document_count": len(multi_doc),
        "multi_document_all_count": sum(item["all_required_source_hit"] for item in multi_doc),
    }


def _structural_checks(items: list[dict]) -> dict:
    return {
        "all_items_at_most_k": all(item["retrieved_count"] <= K for item in items),
        "all_chunk_ids_unique_per_item": all(
            len({chunk["chunk_id"] for chunk in item["retrieved"]}) == item["retrieved_count"]
            for item in items
        ),
        "all_required_source_fields_present": all(
            "required_sources" in item and "all_required_source_hit" in item for item in items
        ),
    }


def _warm_latency(collection, strategy: str, questions: list[dict]) -> dict:
    # One identical warm-up and three identical timed passes. The warm-up is not included in the
    # distribution, and model loading happened before this function was called.
    for item in questions:
        retrieve(item["question"], k=K, rerank=True, strategy=strategy, collection=collection)
    timings = []
    for _ in range(3):
        for item in questions:
            start = time.perf_counter()
            retrieve(item["question"], k=K, rerank=True, strategy=strategy, collection=collection)
            timings.append((time.perf_counter() - start) * 1000)
    return _percentiles(timings)


def _selection(variants: list[dict]) -> dict:
    baseline = next(variant for variant in variants if variant["label"] == "A")
    baseline_all = baseline["summary"]["generation_all"]
    baseline_any = baseline["summary"]["generation_any"]
    baseline_p95 = baseline["warm_latency_ms"]["p95"]
    for variant in variants:
        summary = variant["summary"]
        structural = all(variant["structural_checks"].values())
        variant["selection_qualification"] = {
            "all_required_improvement_items": summary["generation_all"] - baseline_all,
            "any_source_loss_items": baseline_any - summary["generation_any"],
            "all_structural_regressions_pass": structural,
            "warm_p95_within_2x_baseline": (
                baseline_p95 is not None
                and variant["warm_latency_ms"]["p95"] is not None
                and variant["warm_latency_ms"]["p95"] <= baseline_p95 * 2
            ),
        }
        q = variant["selection_qualification"]
        variant["qualifies"] = (
            q["all_required_improvement_items"] >= 3
            and q["any_source_loss_items"] <= 1
            and q["all_structural_regressions_pass"]
            and q["warm_p95_within_2x_baseline"]
        )

    eligible = [
        variant for variant in variants if variant["label"] in {"B", "C"} and variant["qualifies"]
    ]
    if eligible:
        selected = sorted(
            eligible,
            key=lambda variant: (
                -variant["summary"]["generation_all"],
                variant["warm_latency_ms"]["p95"],
                variant["simplicity"],
            ),
        )[0]
        reason = "qualified under the predeclared rule; selected by all-source coverage, p95, then simplicity"
    else:
        selected = baseline
        reason = "neither B nor C qualified; retain baseline A and record the negative result"
    return {
        "selected_variant": selected["label"],
        "provisional": True,
        "serving_configuration_changed": False,
        "candidate_status": "development",
        "reason": reason,
        "rule": {
            "all_required_improvement_items": ">= 3 vs A",
            "any_source_loss_items": "<= 1 vs A",
            "warm_p95": "<= 2x A",
            "structural_regressions": "all pass",
        },
    }


def render_comparison_markdown(result: dict) -> str:
    lines = [
        "# Development retrieval comparison",
        "",
        "Development-only retrieval evidence. No answer model or judge was called; these results do not establish better answers or independent expert validation.",
        "",
        f"**Provisional choice:** {result['selection']['selected_variant']} — {result['selection']['reason']}",
        "",
        "## Variant summary",
        "",
        "| variant | configuration | generation any/all | retrieval any/all | multi-doc all | chunks | embed s | warm p95 ms | qualifies |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in result["variants"]:
        summary = variant["summary"]
        lines.append(
            f"| {variant['label']} | {variant['chunk_target_tokens']}/{variant['chunk_overlap_tokens']} {variant['tokenizer_mode']} + {variant['strategy']} | "
            f"{summary['generation_any']}/{summary['generation_all']} | {summary['retrieval_any']}/{summary['retrieval_all']} | "
            f"{summary['multi_document_all_count']}/{summary['multi_document_count']} | {variant['chunk_count']} | "
            f"{variant['embedding_build_seconds']:.2f} | {variant['warm_latency_ms']['p95']:.2f} | {variant['qualifies']} |"
        )
    lines += [
        "",
        "## Selection and limitations",
        "",
        "The predeclared rule requires at least three additional all-required-source generation items, no more than one lost any-source item, all structural checks passing, and warm p95 no more than twice A. Page-span hits can overstate semantic support; that is checked only in later answer review. A qualified retrieval variant is still only a development candidate.",
        "",
        "## Per-item evidence",
        "",
        "| variant | set | id | required sources | any | all | first relevant rank | retrieved doc/page sequence | truncation (embed/rerank) |",
        "|---|---|---|---|---|---|---:|---|---:|",
    ]
    for variant in result["variants"]:
        for item in variant["per_item"]:
            sequence = ", ".join(
                f"{chunk['doc_id']} p.{chunk['page_start']}-{chunk['page_end']}"
                for chunk in item["retrieved"]
            )
            lines.append(
                f"| {variant['label']} | {item['set']} | {item['id']} | {item['required_sources']} | "
                f"{item['any_required_source_hit']} | {item['all_required_source_hit']} | "
                f"{item['first_relevant_rank'] or 'miss'} | {sequence} | "
                f"{item['embedding_truncated_retrieved']}/{item['reranker_truncated_retrieved']} |"
            )
    return "\n".join(lines) + "\n"


def _assert_output_allowed(output: Path) -> None:
    forbidden = {
        (config.REPORTS_DIR / "retrieval_bench.md").resolve(),
        (config.REPORTS_DIR / "retrieval_bench.json").resolve(),
        (config.REPORTS_DIR / "eval_history.csv").resolve(),
        (config.REPORTS_DIR / "eval_history_v2.csv").resolve(),
    }
    if output.resolve() in forbidden:
        raise ValueError("comparison output cannot overwrite a canonical release report")


def compare(split: str = "dev") -> dict:
    validate_split(split)
    if len(VARIANTS) != 3 or tuple(variant["label"] for variant in VARIANTS) != ("A", "B", "C"):
        raise RuntimeError("comparison must contain exactly variants A, B, and C")

    golden = [
        json.loads(line)
        for line in config.GOLDEN_DEV_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retrieval_questions = json.loads(config.RETRIEVAL_DEV_PATH.read_text(encoding="utf-8"))
    runtime_start = time.perf_counter()
    embedding_model = _get_model()
    cross_encoder = _get_cross_encoder()
    embedding_tokenizer = get_embedding_tokenizer()
    cold_load_seconds = time.perf_counter() - runtime_start
    reranker_tokenizer = cross_encoder.tokenizer
    embedding_limit = _max_length(embedding_model, embedding_tokenizer)
    reranker_limit = _max_length(cross_encoder, reranker_tokenizer)
    stores_root = Path(tempfile.mkdtemp(prefix="regrag_retrieval_comparison_"))

    variants = []
    for variant in VARIANTS:
        tokenizer = None if variant["tokenizer_mode"] == "tiktoken" else embedding_tokenizer
        store_dir = stores_root / variant["label"]
        collection, chunk_count, build_seconds, chunks = _build_isolated_collection(
            variant["chunk_target_tokens"],
            chunk_overlap_tokens=variant["chunk_overlap_tokens"],
            tokenizer=tokenizer,
            root_dir=store_dir,
            collection_name=f"retrieval_{variant['label']}",
        )
        index_payload, pipeline_payload = _identity_payload(variant)
        per_item = []
        for item in golden:
            per_item.append(
                _evaluate_item(
                    item,
                    item_set="golden_dev",
                    collection=collection,
                    strategy=variant["strategy"],
                    embedding_tokenizer=embedding_tokenizer,
                    embedding_limit=embedding_limit,
                    reranker_tokenizer=reranker_tokenizer,
                    reranker_limit=reranker_limit,
                )
            )
        for item in retrieval_questions:
            per_item.append(
                _evaluate_item(
                    item,
                    item_set="retrieval_dev",
                    collection=collection,
                    strategy=variant["strategy"],
                    embedding_tokenizer=embedding_tokenizer,
                    embedding_limit=embedding_limit,
                    reranker_tokenizer=reranker_tokenizer,
                    reranker_limit=reranker_limit,
                )
            )
        golden_items = [item for item in per_item if item["set"] == "golden_dev"]
        retrieval_items = [item for item in per_item if item["set"] == "retrieval_dev"]
        chunk_lengths = [_encode_length(embedding_tokenizer, chunk.text) for chunk in chunks]
        pair_lengths = [length for item in per_item for length in item["reranker_pair_lengths"]]
        variant_result = {
            **variant,
            "store_dir": str(store_dir),
            "collection_name": collection.name,
            "index_identity": hashlib.sha256(
                json.dumps(index_payload, sort_keys=True).encode()
            ).hexdigest(),
            "pipeline_identity": hashlib.sha256(
                json.dumps(pipeline_payload, sort_keys=True).encode()
            ).hexdigest(),
            "index_identity_payload": index_payload,
            "pipeline_identity_payload": pipeline_payload,
            "chunk_count": chunk_count,
            "embedding_build_seconds": build_seconds,
            "cold_load_seconds": cold_load_seconds,
            "embedding_limit": embedding_limit,
            "reranker_limit": reranker_limit,
            "embedding_truncation": {
                "lengths": _percentiles(chunk_lengths),
                "truncated_chunks": sum(length > embedding_limit for length in chunk_lengths),
            },
            "reranker_truncation": {
                "lengths": _percentiles(pair_lengths),
                "truncated_pairs": sum(length > reranker_limit for length in pair_lengths),
            },
            "per_item": per_item,
            "summary": {
                "generation_any": _summarise(golden_items)["any_count"],
                "generation_all": _summarise(golden_items)["all_count"],
                "retrieval_any": _summarise(retrieval_items)["any_count"],
                "retrieval_all": _summarise(retrieval_items)["all_count"],
                "multi_document_all_count": _summarise(golden_items)["multi_document_all_count"],
                "multi_document_count": _summarise(golden_items)["multi_document_count"],
                "generation_mrr_first_relevant": _summarise(golden_items)["mrr_first_relevant"],
                "retrieval_mrr_first_relevant": _summarise(retrieval_items)["mrr_first_relevant"],
            },
            "structural_checks": _structural_checks(per_item),
        }
        variant_result["warm_latency_ms"] = _warm_latency(
            collection, variant["strategy"], retrieval_questions
        )
        variants.append(variant_result)

    result = {
        "schema": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "split": split,
        "question_sets": {
            "golden_dev": {"path": str(config.GOLDEN_DEV_PATH), "count": len(golden)},
            "retrieval_dev": {
                "path": str(config.RETRIEVAL_DEV_PATH),
                "count": len(retrieval_questions),
            },
        },
        "runtime": {
            "cold_load_seconds": cold_load_seconds,
            "hardware": platform.platform(),
            "cpu_count": os.cpu_count(),
            "same_question_order_for_warm_timing": True,
            "warm_timing_passes": 3,
            "warm_timing_warmups": 1,
            "answer_model_called": False,
            "judge_called": False,
            "holdout_accessed": False,
            "serving_store_changed": False,
            "isolated_store_root": str(stores_root),
        },
        "variants": variants,
    }
    result["selection"] = _selection(variants)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _assert_output_allowed(args.output)
    try:
        result = compare(args.split)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = config.REPORTS_DIR / "retrieval_comparison.md"
    markdown_path.write_text(render_comparison_markdown(result), encoding="utf-8", newline="\n")
    print(
        f"wrote {args.output} and {markdown_path}; provisional choice "
        f"{result['selection']['selected_variant']}"
    )


if __name__ == "__main__":
    main()
