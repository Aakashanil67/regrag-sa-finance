"""Eval-driven improvement sweep: chunk size (300/500/800 tokens) x reranking on/off, each
combination measured against the retrieval benchmark, written to reports/improvement_log.md.

Each chunk-size variant gets its own throwaway ChromaDB collection under a separate directory —
the production collection (src.config.CHROMA_DIR / COLLECTION_NAME, built by `python -m
src.store --rebuild`) is never touched by this script and is only rebuilt afterward, by hand, if
a variant other than the current default wins.

This runs the exact retrieval code path production uses (src.retrieve.retrieve, reranking
included) against each experimental collection, rather than a second hand-rolled copy of the
query logic — a separate implementation risks the experiment silently measuring different
behaviour than what actually ships.

    python -m scripts.sweep_chunk_size
"""

import json
import shutil
import time

import chromadb
from chromadb.config import Settings

from src import config
from src.chunking import chunk_document
from src.ingest import extract_elements
from src.retrieve import retrieve
from src.store import _chunk_metadata, embed_texts

SWEEP_CHROMA_DIR = config.ROOT / "chroma_sweep"
CHUNK_SIZES = (300, 500, 800)
K_VALUES = (3, 5, 10)
MAX_K = max(K_VALUES)


def _build_isolated_collection(
    chunk_target_tokens: int,
    *,
    chunk_overlap_tokens: int,
    tokenizer=None,
    root_dir=SWEEP_CHROMA_DIR,
    collection_name: str | None = None,
):
    """Build one isolated corpus collection for a bounded retrieval experiment.

    This is the shared construction path for the historical chunk-size sweep and the
    three-variant comparison. The production Chroma directory is never touched.
    """
    start = time.perf_counter()
    manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    all_chunks = []
    for entry in manifest:
        pdf_path = config.CORPUS_DIR / entry["filename"]
        if not pdf_path.exists():
            continue
        elements = extract_elements(pdf_path)
        all_chunks.extend(
            chunk_document(
                entry["id"],
                elements,
                tokenizer=tokenizer,
                target_tokens=chunk_target_tokens,
                overlap_tokens=chunk_overlap_tokens,
            )
        )

    root_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(root_dir), settings=Settings(anonymized_telemetry=False)
    )
    collection_name = collection_name or f"sweep_{chunk_target_tokens}"
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass  # chromadb raises plain ValueError for "collection doesn't exist" — first run, or after a manual chroma_sweep/ cleanup
    collection = client.create_collection(collection_name)

    embeddings = embed_texts([c.text for c in all_chunks])
    collection.add(
        ids=[c.chunk_hash for c in all_chunks],
        embeddings=embeddings,
        documents=[c.text for c in all_chunks],
        metadatas=[_chunk_metadata(c) for c in all_chunks],
    )
    return collection, len(all_chunks), time.perf_counter() - start, all_chunks


def _build_variant_collection(chunk_target_tokens: int):
    collection, count, _, _ = _build_isolated_collection(
        chunk_target_tokens,
        chunk_overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
    )
    return collection, count


def _is_hit(chunk, expected_doc_id: str, expected_page: int) -> bool:
    return chunk.doc_id == expected_doc_id and chunk.page_start <= expected_page <= chunk.page_end


def _first_hit_rank(results, expected_doc_id: str, expected_page: int):
    for rank, chunk in enumerate(results, start=1):
        if _is_hit(chunk, expected_doc_id, expected_page):
            return rank
    return None


def _bench(collection, retrieval_set: list[dict], rerank: bool) -> dict:
    ranks = []
    for item in retrieval_set:
        results = retrieve(item["question"], k=MAX_K, rerank=rerank, collection=collection)
        ranks.append(_first_hit_rank(results, item["doc_id"], item["page"]))

    hit_rates = {
        k: sum(1 for r in ranks if r is not None and r <= k) / len(ranks) for k in K_VALUES
    }
    mrr = sum(1 / r if r else 0.0 for r in ranks) / len(ranks)
    return {"hit_rates": hit_rates, "mrr": mrr}


def run() -> list[dict]:
    # development data only — a parameter sweep tunes toward whatever set it's run against, so it
    # must never see the sealed holdout (see evals/protocol.json once evals/golden_holdout.jsonl
    # and evals/retrieval_holdout.json exist)
    retrieval_set = json.loads(config.RETRIEVAL_DEV_PATH.read_text(encoding="utf-8"))

    results = []
    for chunk_size in CHUNK_SIZES:
        print(f"building {chunk_size}-token chunk variant...")
        start = time.perf_counter()
        collection, n_chunks = _build_variant_collection(chunk_size)
        build_seconds = time.perf_counter() - start

        for rerank in (False, True):
            label = f"chunk={chunk_size}, rerank={rerank}"
            print(f"  benchmarking {label}...")
            bench = _bench(collection, retrieval_set, rerank=rerank)
            results.append(
                {
                    "chunk_size": chunk_size,
                    "rerank": rerank,
                    "n_chunks": n_chunks,
                    "build_seconds": build_seconds,
                    **bench,
                }
            )
            print(f"    hit-rate@5={bench['hit_rates'][5]:.0%} MRR={bench['mrr']:.3f}")

    return results


def _write_report(results: list[dict]) -> None:
    lines = [
        "# Improvement log: chunk size x reranking sweep",
        "",
        "**Development-set results only** (`evals/retrieval_dev.json`) — this sweep exists to "
        "choose parameters, so it must never touch the sealed holdout. Not release evidence.",
        "",
        "Each row re-chunks and re-embeds the whole corpus at that chunk size into a throwaway "
        "collection (never the production one), then runs the same retrieval benchmark used in "
        "`reports/retrieval_bench.md`.",
        "",
        "| chunk size | rerank | chunks | hit-rate@3 | hit-rate@5 | hit-rate@10 | MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        hr = r["hit_rates"]
        lines.append(
            f"| {r['chunk_size']} | {r['rerank']} | {r['n_chunks']} | {hr[3]:.0%} | "
            f"{hr[5]:.0%} | {hr[10]:.0%} | {r['mrr']:.3f} |"
        )

    best = max(results, key=lambda r: (r["hit_rates"][5], r["mrr"]))
    lines += [
        "",
        f"**Best on hit-rate@5 (tiebreak MRR): chunk size {best['chunk_size']}, "
        f"rerank={best['rerank']}** — hit-rate@5={best['hit_rates'][5]:.0%}, MRR={best['mrr']:.3f}.",
    ]

    (config.REPORTS_DIR / "improvement_log.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\nwrote reports/improvement_log.md")
    print(f"best: chunk_size={best['chunk_size']} rerank={best['rerank']}")


def main() -> None:
    try:
        results = run()
        _write_report(results)
    finally:
        try:
            shutil.rmtree(SWEEP_CHROMA_DIR)
        except OSError as exc:
            # seen on Windows: chromadb's sqlite connection outlives the PersistentClient object
            # (no explicit .close()), so the file is still locked here — silently swallowing this
            # would leave 40+MB of scratch data with no indication anything needs cleaning up.
            print(f"warning: couldn't remove {SWEEP_CHROMA_DIR} ({exc}); delete it by hand")


if __name__ == "__main__":
    main()
