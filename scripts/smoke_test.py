"""Runs a fixed set of real questions end to end (retrieval + live LLM call) and writes a
transcript for manual review, the point isn't automated pass/fail, it's a human spot-checking
that citations actually point at the right pages before any eval harness gets built on top of
this. Every question is logged through obslog.timed_answer, so it also seeds the SQLite log and
the response cache with real traffic.

    python -m scripts.smoke_test
"""

from src.config import REPORTS_DIR
from src.obslog import timed_answer

QUESTIONS = [
    "What is the expected credit loss model under IFRS 9?",
    "How many days does a debt counsellor have to file a clearance certificate with credit bureaus?",
    "What does Directive D3/2023 say about classifying impairments as general or specific?",
    "What seven categories does the BCBS operational resilience framework cover?",
    "What is the National Credit Act's stated purpose?",
    "Under which Act was the Conduct Standard for OTC derivative providers published?",
    "What did the 2014 Retail Distribution Review propose about intermediary remuneration?",
    "What is the current South African repo rate?",  # unanswerable, not in this corpus
    "What are the JSE's main board listing requirements?",  # unanswerable, not in this corpus
    "How many climate-related disclosure templates does Guidance Note 3/2025 include?",
]


def run() -> None:
    lines = ["# Smoke test transcript", ""]
    for i, question in enumerate(QUESTIONS, start=1):
        print(f"[{i}/{len(QUESTIONS)}] {question}")
        timed = timed_answer(question)
        result = timed.result

        lines.append(f"## {i}. {question}")
        lines.append("")
        lines.append(
            f"**Refused:** {result.refused} | **Latency:** {timed.latency_ms:.0f}ms | "
            f"**Cost:** ${result.llm_response.cost_usd:.4f} | **Model:** {result.llm_response.model}"
        )
        lines.append("")
        lines.append(f"**Answer:**\n\n{result.answer}")
        lines.append("")
        if result.citations:
            verified = sum(1 for c in result.citations if c.verified)
            lines.append(f"**Citations:** {len(result.citations)} total, {verified} verified")
            for c in result.citations:
                mark = "OK" if c.verified else "UNVERIFIED"
                lines.append(f"- [{mark}] {c.doc_id}, p.{c.page}")
        else:
            lines.append("**Citations:** none")
        lines.append("")
        lines.append("---")
        lines.append("")

    (REPORTS_DIR / "smoke_test.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"\nwrote reports/smoke_test.md ({len(QUESTIONS)} questions)")


if __name__ == "__main__":
    run()
