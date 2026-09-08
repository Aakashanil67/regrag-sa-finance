"""RAGAS evaluation over the answerable items (factual + multi-doc) in evals/golden_dev.jsonl.

Runs the real RAG pipeline on each question — real retrieval, real Claude answer — then scores
that output against the golden reference answer with four RAGAS metrics.

Unanswerable items are excluded on purpose: all four metrics assume the system tried to answer
from context, which doesn't apply to a correct refusal. Refusal correctness (and citation
presence) is checked by evals/test_snapshot_integrity.py instead, not by these metrics.

What each metric actually measures, briefly:
- Faithfulness: of the individual claims in the answer, what fraction are actually supported by
  the retrieved context? Catches hallucination — an answer can be perfectly faithful and still be
  *wrong* if the retrieved context itself was wrong or incomplete, which is why this is not the
  same thing as accuracy.
- Answer relevancy: does the answer actually address the question asked, judged independently of
  whether it's grounded in context. Punishes padding, hedging, or answering an adjacent question.
- Context precision: of the chunks retrieval returned, how many were actually relevant to
  answering correctly? Low precision means retrieval is pulling in noise alongside the useful
  chunks.
- Context recall: of what the reference answer actually needed, how much did the retrieved
  context cover? Low recall means retrieval is missing content the correct answer depends on —
  no amount of prompt engineering downstream fixes that.

    python -m evals.run_ragas
"""

import asyncio
import json

from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from evals._ragas_judge import build_judge
from src.config import GOLDEN_DEV_PATH
from src.rag import answer_question

_SCORED_TYPES = {"factual", "multi-doc"}
_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


async def _score_item(item: dict, metrics: dict) -> dict | None:
    result = answer_question(item["question"])
    if result.refused:
        # a "factual"/"multi-doc" golden item can still hit a real retrieval gap and correctly
        # trigger rag.py's own refusal rule — scoring "I don't have a source for that." against
        # metrics built for a substantive, grounded answer produces near-zero noise, not signal.
        # Confirmed live: several golden items scored 0.00 across all four metrics on the first
        # run, and every one of them turned out to be a refusal, not a bad answer.
        return None

    contexts = [c.text for c in result.retrieved_chunks]
    reference = item["reference_answer"]

    faithfulness, answer_relevancy, context_precision, context_recall = await asyncio.gather(
        metrics["faithfulness"].ascore(
            user_input=item["question"], response=result.answer, retrieved_contexts=contexts
        ),
        metrics["answer_relevancy"].ascore(user_input=item["question"], response=result.answer),
        metrics["context_precision"].ascore(
            user_input=item["question"], reference=reference, retrieved_contexts=contexts
        ),
        metrics["context_recall"].ascore(
            user_input=item["question"], retrieved_contexts=contexts, reference=reference
        ),
    )
    return {
        "id": item["id"],
        "question": item["question"],
        "faithfulness": faithfulness.value,
        "answer_relevancy": answer_relevancy.value,
        "context_precision": context_precision.value,
        "context_recall": context_recall.value,
    }


async def run() -> dict:
    golden = [json.loads(line) for line in GOLDEN_DEV_PATH.read_text(encoding="utf-8").splitlines()]
    scored_items = [item for item in golden if item["type"] in _SCORED_TYPES]

    llm, embeddings = build_judge()
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    per_item = []
    refusals = []
    failures = []
    for i, item in enumerate(scored_items, start=1):
        print(f"[{i}/{len(scored_items)}] {item['id']}: {item['question'][:70]}")
        try:
            scored = await _score_item(item, metrics)
        except Exception as exc:
            # a single item's judge call failing (seen live: Claude's structured-output JSON
            # got cut off mid-response on an answer with many claims to verify) must not throw
            # away every dollar already spent scoring the rest of the set — record it and move on
            print(f"  FAILED: {exc}")
            failures.append({"id": item["id"], "question": item["question"], "error": str(exc)})
            continue

        if scored is None:
            print("  refused (excluded from RAGAS means, see reports/failure_analysis.md)")
            refusals.append({"id": item["id"], "question": item["question"]})
        else:
            per_item.append(scored)

    if not per_item:
        raise RuntimeError(f"all {len(scored_items)} items failed or refused — nothing to report")

    means = {name: sum(row[name] for row in per_item) / len(per_item) for name in _METRIC_NAMES}
    return {
        "per_item": per_item,
        "means": means,
        "n_scored": len(per_item),
        "n_attempted": len(scored_items),
        "refusals": refusals,
        "failures": failures,
    }


def main() -> None:
    """Ad-hoc, uncommitted RAGAS scoring for local iteration — prints means and per-item scores to
    the terminal only. Does not touch reports/eval_summary.md or reports/eval_history.csv: those
    canonical files are owned exclusively by `python -m evals.run_release`, which computes the
    same RAGAS metrics denominator-aware alongside the citation-contract structural metrics and
    only promotes them on a complete run. Use this module directly only to sanity-check a change
    quickly; use run_release for anything that should be recorded.
    """
    result = asyncio.run(run())
    means = result["means"]
    print(
        f"\nfaithfulness={means['faithfulness']:.3f} answer_relevancy={means['answer_relevancy']:.3f} "
        f"context_precision={means['context_precision']:.3f} context_recall={means['context_recall']:.3f}"
    )
    if result["refusals"]:
        print(f"{len(result['refusals'])} item(s) refused — excluded from means")
    if result["failures"]:
        print(f"{len(result['failures'])} item(s) failed to score")
    print(
        f"scored {result['n_scored']}/{result['n_attempted']} (not recorded — see module docstring)"
    )


if __name__ == "__main__":
    main()
