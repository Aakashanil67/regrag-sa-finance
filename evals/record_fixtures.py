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

import asyncio
import json

from ragas.metrics.collections import Faithfulness

from evals._ragas_judge import build_judge
from evals.snapshot import CI_SUBSET_IDS, compute_snapshot_metadata
from src.config import EVALS_DIR, GOLDEN_DEV_PATH
from src.rag import answer_question

FIXTURES_PATH = EVALS_DIR / "fixtures" / "ci_subset.json"


async def record_one(item: dict, faithfulness) -> dict:
    result = answer_question(item["question"])
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
        score = await faithfulness.ascore(
            user_input=item["question"], response=result.answer, retrieved_contexts=contexts
        )
        record["faithfulness"] = score.value
    return record


async def run() -> list[dict]:
    golden = {
        item["id"]: item
        for item in (
            json.loads(line) for line in GOLDEN_DEV_PATH.read_text(encoding="utf-8").splitlines()
        )
    }
    items = [golden[item_id] for item_id in CI_SUBSET_IDS]

    llm, _ = build_judge()
    faithfulness = Faithfulness(llm=llm)

    records = []
    for i, item in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {item['id']}: {item['question'][:70]}")
        records.append(await record_one(item, faithfulness))
    return records


def main() -> None:
    records = asyncio.run(run())
    fixture = {"metadata": compute_snapshot_metadata(), "records": records}
    FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES_PATH.write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {FIXTURES_PATH.relative_to(EVALS_DIR.parent)} ({len(records)} records)")


if __name__ == "__main__":
    main()
