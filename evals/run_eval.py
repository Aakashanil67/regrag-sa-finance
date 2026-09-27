"""Run the pipeline over a question split and save one artifact per run.

    python -m evals.run_eval --split dev --config wordpiece --label wordpiece --max-usd 0.1
    python -m evals.run_eval --split dev --config default --label closedbook --closed-book --max-usd 0.2
    python -m evals.run_eval --split test --config default --label final --max-usd 0.15
    python -m evals.run_eval --split dev --config default --label smoke --limit 2 --max-usd 0.01 --scratch

--scratch prints the summary and saves nothing. Test runs need evals/protocol_test.json to carry a frozen_pipeline, the `default` config must
match it, and an existing test artifact is never overwritten.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from evals import configs
from evals.metrics import ci_text, compute, ratio_text

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "reports" / "runs"
HISTORY_CSV = ROOT / "reports" / "eval_runs.csv"
SPLITS = {
    "dev": ROOT / "evals" / "questions_dev.jsonl",
    "test": ROOT / "evals" / "questions_test.jsonl",
}
PROTOCOL = ROOT / "evals" / "protocol_test.json"
# (input, output) tokens per question: ~4.5k of context for RAG; closed-book sends the question only.
TOKENS_PER_ITEM = {"rag": (4500, 350), "closed_book": (150, 350)}
CLOSED_BOOK_SYSTEM = (
    "You answer questions about South African financial regulation from your own knowledge. "
    'Answer in at most three sentences. If you are not sure, reply with exactly: "{refusal}"'
)
HISTORY_FIELDS = [
    "timestamp",
    "run_id",
    "split",
    "label",
    "config",
    "closed_book",
    "n",
    "answer_rate",
    "refusal_recall",
    "task_outcome",
    "raw_citation_precision",
    "retrieval_any_hit",
    "retrieval_all_hit",
    "cost_usd",
]


def load_items(split: str) -> list[dict]:
    text = SPLITS[split].read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate_usd(n_items: int, kind: str) -> float | None:
    from src.config import PRICING_PER_MILLION_TOKENS
    from src.llm import effective_llm_settings

    model = effective_llm_settings().model
    if model not in PRICING_PER_MILLION_TOKENS:
        return None
    rate_in, rate_out = PRICING_PER_MILLION_TOKENS[model]
    tokens_in, tokens_out = TOKENS_PER_ITEM[kind]
    return n_items * (tokens_in * rate_in + tokens_out * rate_out) / 1_000_000


def check_test_run(label: str, config: str, closed_book: bool) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if _sha256(SPLITS["test"]) != protocol["sha256"]:
        sys.exit("the test questions changed after they were sealed")
    if not protocol.get("frozen_pipeline"):
        sys.exit("pipeline not frozen: set frozen_pipeline in evals/protocol_test.json first")
    if (RUNS_DIR / f"test-{label}.json").exists():
        sys.exit(f"reports/runs/test-{label}.json exists; each test label runs once")
    if config == "default" and not closed_book:
        from src.config import RETRIEVAL_K
        from src.provenance import pipeline_fingerprint

        if pipeline_fingerprint(k=RETRIEVAL_K) != protocol["frozen_pipeline"]:
            sys.exit("the current pipeline is not the frozen one")


def rag_record(item: dict) -> dict:
    from src.rag import answer_question

    res = answer_question(item["question"])
    return {
        "answer": res.answer,
        "served_answer": None if res.refused else res.answer,
        "raw_model_output": res.llm_response.text or None,
        "first_raw_output": getattr(res, "first_raw_output", None),
        "repair_attempted": getattr(res, "repair_attempted", False),
        "refused": res.refused,
        "refusal_reason": res.refusal_reason.value if res.refusal_reason else None,
        "citations": [
            {
                "doc_id": c.doc_id,
                "page": c.page,
                "verified": c.verified,
                "section_ref": c.section_ref,
            }
            for c in res.citations
        ],
        "contexts": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section": c.section,
                "text": c.text,
            }
            for c in res.retrieved_chunks
        ],
        "formatted_context": res.formatted_context,
        "source_notices": [{"kind": n.kind, "text": n.text} for n in res.source_notices],
        "usage": {
            "input_tokens": res.llm_response.input_tokens,
            "output_tokens": res.llm_response.output_tokens,
            "cost_usd": res.llm_response.cost_usd,
        },
    }


def closed_book_record(item: dict) -> dict:
    from src.llm import complete
    from src.rag import INSUFFICIENT_CONTEXT_PHRASE

    res = complete(
        system=CLOSED_BOOK_SYSTEM.format(refusal=INSUFFICIENT_CONTEXT_PHRASE), user=item["question"]
    )
    refused = INSUFFICIENT_CONTEXT_PHRASE.lower() in res.text.lower()
    return {
        "answer": res.text.strip(),
        "served_answer": None if refused else res.text.strip(),
        "raw_model_output": res.text,
        "refused": refused,
        "refusal_reason": "model_refusal" if refused else None,
        "citations": [],
        "contexts": [],
        "usage": {
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "cost_usd": res.cost_usd,
        },
    }


def failed_record(exc: Exception) -> dict:
    return {
        "error": f"{type(exc).__name__}: {exc}",
        "refused": True,
        "refusal_reason": "error",
        "citations": [],
        "contexts": [],
        "raw_model_output": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    }


def render_markdown(artifact: dict) -> str:
    m = artifact["metrics"]
    rows = [
        ("Answerable questions answered", m["answer_rate"]),
        ("Unanswerable questions refused", m["refusal_recall"]),
        ("Correct outcome (answered or correctly refused)", m["task_outcome"]),
        ("Retrieval: an evidence page in the top k", m["retrieval_any_hit"]),
        ("Retrieval: every evidence document in the top k", m["retrieval_all_hit"]),
    ]
    mode = " (closed book)" if artifact["closed_book"] else ""
    lines = [
        f"# {artifact['run_id']}",
        "",
        f"Config `{artifact['config']}`{mode}, {m['n']} questions, {artifact['created_at'][:10]}, cost ${m['cost_usd']:.2f}.",
        "",
        "| metric | result | 95% Wilson CI |",
        "|---|---|---|",
        *[f"| {name} | {ratio_text(r)} | {ci_text(r)} |" for name, r in rows],
        "",
        f"Model citations pointing at a page it was shown: {ratio_text(m['raw_citation_precision'])}. "
        f"Unverified citations served: {m['served_unverified_citations']}. Repairs attempted: {m['repairs']}.",
        f"Refusal reasons: {json.dumps(m['refusal_reasons'], sort_keys=True)}. Failed items: {m['failed_items']}.",
        "",
    ]
    return "\n".join(lines)


def append_history(artifact: dict) -> None:
    m = artifact["metrics"]
    row = {
        "timestamp": artifact["created_at"],
        "run_id": artifact["run_id"],
        "split": artifact["split"],
        "label": artifact["label"],
        "config": artifact["config"],
        "closed_book": artifact["closed_book"],
        "n": m["n"],
        "cost_usd": m["cost_usd"],
        **{
            k: m[k]["rate"]
            for k in (
                "answer_rate",
                "refusal_recall",
                "task_outcome",
                "raw_citation_precision",
                "retrieval_any_hit",
                "retrieval_all_hit",
            )
        },
    }
    new = not HISTORY_CSV.exists()
    with HISTORY_CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--closed-book", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-usd", type=float)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scratch", action="store_true", help="print the summary, save nothing")
    args = ap.parse_args(argv)

    configs.apply(args.config)
    items = load_items(args.split)
    if args.limit:
        items = items[: args.limit]
    if args.split == "test":
        check_test_run(args.label, args.config, args.closed_book)
    kind = "closed_book" if args.closed_book else "rag"
    estimate = estimate_usd(len(items), kind)
    if args.dry_run:
        cost = "no price for the model" if estimate is None else f"estimated ${estimate:.2f}"
        print(f"dry run: {len(items)} {args.split} items, config {args.config}, {kind}, {cost}")
        return
    if estimate is None:
        sys.exit("no price for the model in PRICING_PER_MILLION_TOKENS; add it before spending")
    if args.max_usd is None or estimate > args.max_usd:
        sys.exit(f"estimated ${estimate:.2f}; pass --max-usd of at least that")

    records: list[dict] = []
    spent = 0.0
    for n, item in enumerate(items, 1):
        start = time.perf_counter()
        try:
            record = closed_book_record(item) if args.closed_book else rag_record(item)
        except Exception as exc:  # noqa: BLE001 - a failed item is kept and counted, never dropped
            record = failed_record(exc)
        base = {k: item.get(k) for k in ("id", "type", "question", "reference_answer", "evidence")}
        records.append(
            {**base, **record, "latency_ms": round((time.perf_counter() - start) * 1000)}
        )
        spent += record["usage"]["cost_usd"]
        print(
            f"[{n}/{len(items)}] {item['id']} {'refused' if record['refused'] else 'answered'} ${spent:.4f}",
            flush=True,
        )
        if spent > args.max_usd:
            print(f"stopping: spent ${spent:.4f}, over --max-usd {args.max_usd}")
            break

    from src.config import RETRIEVAL_K, effective_settings
    from src.provenance import pipeline_fingerprint

    artifact = {
        "schema": 3,
        "run_id": f"{args.split}-{args.label}",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": args.split,
        "label": args.label,
        "config": args.config,
        "closed_book": args.closed_book,
        "settings": effective_settings(),
        "pipeline_fingerprint": None if args.closed_book else pipeline_fingerprint(k=RETRIEVAL_K),
        "questions_sha256": _sha256(SPLITS[args.split]),
        "complete": len(records) == len(items) and not any(r.get("error") for r in records),
        "metrics": compute(records),
        "items": records,
    }
    if args.scratch:
        print(render_markdown(artifact))
        print(f"cost ${artifact['metrics']['cost_usd']:.4f}")
        return
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{artifact['run_id']}.json"
    out.write_text(json.dumps(artifact, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(artifact), encoding="utf-8")
    append_history(artifact)
    print(render_markdown(artifact))
    print(f"cost ${artifact['metrics']['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
