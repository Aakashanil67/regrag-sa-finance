"""Label every answer in a saved run with the rubric the human reviewer also uses.

    python -m evals.judge reports/runs/dev-wordpiece.json --judge qwen

Judges are local Ollama models (free): qwen = qwen2.5:7b, llama = llama3.1:8b.
correctness: correct | partial | incorrect (answer vs the reference answer)
support: supported | partial | unsupported (answer vs the saved context; RAG runs only)
Refused answerable items are labelled "refused" without a model call.
"""

import argparse
import json
import re
from pathlib import Path

JUDGE_MODELS = {"qwen": "qwen2.5:7b", "llama": "llama3.1:8b"}
CORRECTNESS = ("correct", "partial", "incorrect")
SUPPORT = ("supported", "partial", "unsupported")
SYSTEM = "You grade answers about South African financial regulation. Reply with JSON only."
PROMPT_CORRECTNESS = """Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Label the candidate against the reference.
correct: states the reference's key facts and nothing that contradicts it.
partial: some key facts right, others missing or wrong.
incorrect: wrong, contradicts the reference, or answers a different question.
Reply as {{"label": "correct|partial|incorrect", "reason": "<one sentence>"}}"""
PROMPT_SUPPORT = """Context the answer was written from:
{context}

Answer: {answer}

supported: every sentence of the answer is stated in the context.
partial: some sentences are, some are not.
unsupported: the main claim is not in the context.
Reply as {{"label": "supported|partial|unsupported", "reason": "<one sentence>"}}"""
_JSON = re.compile(r"\{.*\}", re.S)


def parse_label(text: str, allowed: tuple[str, ...]) -> tuple[str, str]:
    match = _JSON.search(text or "")
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    label = data.get("label")
    return (label if label in allowed else "unparsed"), str(data.get("reason", ""))[:300]


def judge_items(run: dict, ask) -> list[dict]:
    """`ask(prompt) -> text`. Returns one label row per item."""
    out = []
    for item in run["items"]:
        row = {"id": item["id"], "type": item["type"]}
        if item["type"] == "unanswerable":
            row["correctness"] = "refused" if item["refused"] else "answered_unanswerable"
        elif item["refused"]:
            row["correctness"] = "refused"
        else:
            text = ask(
                PROMPT_CORRECTNESS.format(
                    question=item["question"],
                    reference=item["reference_answer"],
                    answer=item["served_answer"],
                )
            )
            row["correctness"], row["correctness_reason"] = parse_label(text, CORRECTNESS)
            if not run["closed_book"]:
                text = ask(
                    PROMPT_SUPPORT.format(
                        context=item["formatted_context"], answer=item["served_answer"]
                    )
                )
                row["support"], row["support_reason"] = parse_label(text, SUPPORT)
        out.append(row)
    return out


def summarise(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    count = lambda field, value: sum(1 for r in answerable if r.get(field) == value)  # noqa: E731
    return {
        "answerable": len(answerable),
        **{label: count("correctness", label) for label in (*CORRECTNESS, "refused", "unparsed")},
        "support": {label: count("support", label) for label in (*SUPPORT, "unparsed")},
        "answered_unanswerable": sum(
            1 for r in rows if r.get("correctness") == "answered_unanswerable"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--judge", choices=sorted(JUDGE_MODELS), required=True)
    args = ap.parse_args(argv)
    run_path = Path(args.run)
    run = json.loads(run_path.read_text(encoding="utf-8"))

    import os

    import httpx

    host = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"

    def ask(prompt: str) -> str:
        # Called directly rather than through src.llm: an 8B model on a 4 GB card can exceed
        # that client's 120 s timeout on long contexts, and format=json keeps labels parseable.
        for attempt in range(3):
            try:
                r = httpx.post(
                    f"{host}/api/chat",
                    json={
                        "model": JUDGE_MODELS[args.judge],
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                        "format": "json",
                        "options": {"num_predict": 200, "temperature": 0, "num_ctx": 8192},
                    },
                    timeout=900.0,
                )
                r.raise_for_status()
                return r.json()["message"]["content"]
            except httpx.TransportError:
                if attempt == 2:
                    return ""
        return ""

    rows = judge_items(run, ask)
    result = {
        "run": run_path.name,
        "judge": args.judge,
        "model": JUDGE_MODELS[args.judge],
        "summary": summarise(rows),
        "items": rows,
    }
    out = run_path.with_name(f"{run_path.stem}.judge-{args.judge}.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
