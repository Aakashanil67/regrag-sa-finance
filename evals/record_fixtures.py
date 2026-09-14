"""Records real API responses for a fixed 10-item subset of the golden set into
evals/fixtures/ci_subset.json — this is what evals/test_regression.py checks against in CI, so
the regression gate catches an actual behaviour change without needing an API key or spending
money on every push.

Re-run this manually (locally, or via a workflow_dispatch job) whenever rag.py's prompt or logic
changes enough that the recorded answers should legitimately change. Committing the refreshed
fixture is then the record of "we intentionally moved the baseline," same as updating a snapshot
test.

    python -m evals.record_fixtures
"""

import argparse
import asyncio
import json
from pathlib import Path

from ragas.metrics.collections import Faithfulness

from evals._ragas_judge import build_judge
from evals.sealed_run import CostBudget, estimate_cost_usd
from evals.snapshot import CI_SUBSET_IDS, compute_snapshot_metadata
from src.config import DEFAULT_ANTHROPIC_MODEL, EVALS_DIR, GOLDEN_DEV_PATH, RAG_MAX_ANSWER_TOKENS
from src.rag import answer_question

FIXTURES_PATH = EVALS_DIR / "fixtures" / "ci_subset.json"
DEFAULT_OUTPUT_PATH = EVALS_DIR / "fixtures" / "ci_subset_v2.json"
GENERATION_INPUT_TOKENS = 8192
JUDGE_INPUT_TOKENS = 4096
JUDGE_OUTPUT_TOKENS = 512


def estimate_recording_cost(item_count: int, model: str) -> float:
    """Conservative generation plus one Faithfulness judge estimate per snapshot item."""
    generation = estimate_cost_usd(model, GENERATION_INPUT_TOKENS, RAG_MAX_ANSWER_TOKENS)
    judge = estimate_cost_usd(model, JUDGE_INPUT_TOKENS, JUDGE_OUTPUT_TOKENS)
    return item_count * (generation + judge)


def validate_output_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    if output_path.resolve() == FIXTURES_PATH.resolve():
        raise ValueError("historical CI snapshot path is immutable; choose a new output path")
    return output_path


async def record_one(
    item: dict,
    faithfulness,
    *,
    budget: CostBudget,
    generation_estimate_usd: float,
    judge_estimate_usd: float,
) -> dict:
    budget.reserve(generation_estimate_usd)
    result = answer_question(item["question"])
    budget.record_actual(result.llm_response.cost_usd)
    record = {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "answer": result.answer,
        "refused": result.refused,
        "citations": [
            {"doc_id": c.doc_id, "page": c.page, "verified": c.verified} for c in result.citations
        ],
    }
    if item["type"] != "unanswerable" and not result.refused:
        # same scoping rule as run_ragas.py: an answerable-type item that hit a real retrieval
        # gap and correctly refused isn't a faithfulness failure, it's a different, already-
        # tracked failure mode — scoring "I don't have a source for that." against a metric built
        # for a substantive answer produces noise, not a meaningful regression signal.
        contexts = [c.text for c in result.retrieved_chunks]
        budget.reserve(judge_estimate_usd)
        score = await faithfulness.ascore(
            user_input=item["question"], response=result.answer, retrieved_contexts=contexts
        )
        record["faithfulness"] = score.value
    return record


async def run(*, output_path: Path, max_cost_usd: float | None) -> list[dict]:
    if max_cost_usd is None or max_cost_usd <= 0:
        raise ValueError("snapshot recording requires a positive max_cost_usd cap")
    validate_output_path(output_path)
    golden = {
        item["id"]: item
        for item in (
            json.loads(line) for line in GOLDEN_DEV_PATH.read_text(encoding="utf-8").splitlines()
        )
    }
    items = [golden[item_id] for item_id in CI_SUBSET_IDS]

    generation_estimate_usd = estimate_cost_usd(
        DEFAULT_ANTHROPIC_MODEL, GENERATION_INPUT_TOKENS, RAG_MAX_ANSWER_TOKENS
    )
    judge_estimate_usd = estimate_cost_usd(
        DEFAULT_ANTHROPIC_MODEL, JUDGE_INPUT_TOKENS, JUDGE_OUTPUT_TOKENS
    )
    budget = CostBudget(max_cost_usd)
    llm, _ = build_judge()
    faithfulness = Faithfulness(llm=llm)

    records = []
    for i, item in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {item['id']}: {item['question'][:70]}")
        records.append(
            await record_one(
                item,
                faithfulness,
                budget=budget,
                generation_estimate_usd=generation_estimate_usd,
                judge_estimate_usd=judge_estimate_usd,
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    args = parser.parse_args()
    output_path = validate_output_path(args.output)
    records = asyncio.run(run(output_path=output_path, max_cost_usd=args.max_cost_usd))
    fixture = {"metadata": compute_snapshot_metadata(), "records": records}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {output_path.relative_to(EVALS_DIR.parent)} ({len(records)} records)")


if __name__ == "__main__":
    main()
