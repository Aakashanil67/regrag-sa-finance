"""Agent vs. plain RAG, on the multi-doc golden items specifically — not the whole golden set.

src/agent.py exists to fix one diagnosed failure class (reports/failure_analysis.md: g37 and g44,
where a two-document comparison question under-retrieves whichever named document has fewer
chunks). The multi-doc items are exactly the golden-set items shaped like that, so they're the
honest test of whether the extra requery step earns its cost — running the full 45-item golden set
through both pipelines would spend roughly double the API budget to re-confirm what the other 35
single-document items already show works fine without an agent.

    python -m scripts.agent_eval
"""

import json
import time

from src.agent import answer_question as agent_answer
from src.config import GOLDEN_DEV_PATH, REPORTS_DIR
from src.rag import answer_question as plain_answer


def run() -> list[dict]:
    golden = [json.loads(line) for line in GOLDEN_DEV_PATH.read_text(encoding="utf-8").splitlines()]
    multi_doc = [item for item in golden if item["type"] == "multi-doc"]

    rows = []
    for item in multi_doc:
        print(f"{item['id']}: {item['question'][:70]}")

        start = time.perf_counter()
        plain = plain_answer(item["question"])
        plain_latency_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        agentic = agent_answer(item["question"])
        agent_latency_ms = (time.perf_counter() - start) * 1000

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "plain_refused": plain.refused,
                "agent_refused": agentic.refused,
                "agent_requeries": sum(1 for s in agentic.steps if s.action == "requery"),
                "plain_cost_usd": plain.llm_response.cost_usd,
                "agent_cost_usd": agentic.total_cost_usd,
                "plain_latency_ms": plain_latency_ms,
                "agent_latency_ms": agent_latency_ms,
                "fixed_a_refusal": plain.refused and not agentic.refused,
                "introduced_a_refusal": (not plain.refused) and agentic.refused,
            }
        )
    return rows


def write_report(rows: list[dict]) -> None:
    fixed = sum(1 for r in rows if r["fixed_a_refusal"])
    broke = sum(1 for r in rows if r["introduced_a_refusal"])
    mean_cost_ratio = sum(
        r["agent_cost_usd"] / r["plain_cost_usd"] for r in rows if r["plain_cost_usd"]
    ) / len([r for r in rows if r["plain_cost_usd"]])
    mean_latency_ratio = sum(r["agent_latency_ms"] / r["plain_latency_ms"] for r in rows) / len(
        rows
    )

    lines = [
        "# Agent vs. plain RAG — multi-document comparison questions",
        "",
        f"All {len(rows)} `multi-doc`-type golden items, run through both `rag.answer_question` "
        "(single retrieval, k=5) and `agent.answer_question` (retrieve, decide, requery once if a "
        "named document is missing, then answer). Restricted to this subset rather than the full "
        "45-item golden set on purpose — this is the failure class the agent was built to fix "
        "(`reports/failure_analysis.md`, items g37 and g44), and the other 35 single-document "
        "items already work without it, so running them again would only spend API budget "
        "re-confirming that reranked single-shot retrieval is enough for a question that doesn't "
        "name two documents.",
        "",
        f"**Refusals fixed: {fixed}. Refusals introduced: {broke}.**",
        f"**Mean cost ratio (agent / plain): {mean_cost_ratio:.2f}x. "
        f"Mean latency ratio: {mean_latency_ratio:.2f}x.**",
        "",
        "| id | question | plain refused | agent refused | requeries | cost ratio | latency ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cost_ratio = (
            r["agent_cost_usd"] / r["plain_cost_usd"] if r["plain_cost_usd"] else float("nan")
        )
        latency_ratio = r["agent_latency_ms"] / r["plain_latency_ms"]
        lines.append(
            f"| {r['id']} | {r['question']} | {r['plain_refused']} | {r['agent_refused']} | "
            f"{r['agent_requeries']} | {cost_ratio:.2f}x | {latency_ratio:.2f}x |"
        )

    (REPORTS_DIR / "agent_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote reports/agent_eval.md — fixed {fixed}, broke {broke}")


if __name__ == "__main__":
    write_report(run())
