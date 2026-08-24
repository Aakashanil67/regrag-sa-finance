"""Retrieval-only benchmark: hit-rate@k and MRR against a hand-checked question set.

Hit-rate@k asks "is the right document in the top k results at all" — the metric a user
experiences directly, since the RAG layer only ever sees the top k chunks retrieve() returns.
MRR (mean reciprocal rank) asks "how far down the list is it" — a system that always ranks the
right answer 1st scores 1.0, always 3rd scores 0.33 — so two systems with identical hit-rate@10
can still be told apart by how much of the ranking work retrieval is doing versus leaving to luck.

A "hit" means the retrieved chunk's doc_id matches AND the expected page falls inside that
chunk's [page_start, page_end] span — matching the doc alone would call a hit on a chunk from
page 40 of a document when the question is about page 1.
"""

import json

from src.config import REPORTS_DIR
from src.retrieve import RetrievedChunk, retrieve

RETRIEVAL_SET_PATH = REPORTS_DIR.parent / "evals" / "retrieval_set.json"
K_VALUES = (3, 5, 10)
MAX_K = max(K_VALUES)


def _is_hit(chunk: RetrievedChunk, expected_doc_id: str, expected_page: int) -> bool:
    return chunk.doc_id == expected_doc_id and chunk.page_start <= expected_page <= chunk.page_end


def _first_hit_rank(
    results: list[RetrievedChunk], expected_doc_id: str, expected_page: int
) -> int | None:
    for rank, chunk in enumerate(results, start=1):
        if _is_hit(chunk, expected_doc_id, expected_page):
            return rank
    return None


def run_benchmark() -> dict:
    questions = json.loads(RETRIEVAL_SET_PATH.read_text(encoding="utf-8"))

    per_question = []
    for item in questions:
        results = retrieve(item["question"], k=MAX_K)
        rank = _first_hit_rank(results, item["doc_id"], item["page"])
        per_question.append({**item, "first_hit_rank": rank})

    hit_rates = {
        k: sum(
            1 for q in per_question if q["first_hit_rank"] is not None and q["first_hit_rank"] <= k
        )
        / len(per_question)
        for k in K_VALUES
    }
    mrr = sum(1 / q["first_hit_rank"] if q["first_hit_rank"] else 0.0 for q in per_question) / len(
        per_question
    )

    return {"per_question": per_question, "hit_rates": hit_rates, "mrr": mrr}


def write_report(results: dict) -> None:
    lines = [
        "# Retrieval benchmark",
        "",
        "**Hit-rate@k**: fraction of questions where the source document/page appears anywhere in "
        "the top k retrieved chunks — what a user actually experiences, since the RAG layer only "
        "sees the top k.",
        "",
        "**MRR** (mean reciprocal rank): averages 1/rank of the first correct chunk across all "
        f"{MAX_K} retrieved results — rewards ranking the right answer 1st over merely including "
        "it somewhere in the list.",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k in K_VALUES:
        lines.append(f"| hit-rate@{k} | {results['hit_rates'][k]:.0%} |")
    lines.append(f"| MRR | {results['mrr']:.3f} |")

    lines += [
        "",
        "## Per-question results",
        "",
        "| id | question | expected | first hit rank |",
        "|---|---|---|---|",
    ]
    for q in results["per_question"]:
        rank = q["first_hit_rank"] if q["first_hit_rank"] else f"miss (not in top {MAX_K})"
        lines.append(f"| {q['id']} | {q['question']} | {q['doc_id']} p.{q['page']} | {rank} |")

    (REPORTS_DIR / "retrieval_bench.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"wrote reports/retrieval_bench.md — hit-rate@5={results['hit_rates'][5]:.0%}, MRR={results['mrr']:.3f}"
    )


if __name__ == "__main__":
    write_report(run_benchmark())
