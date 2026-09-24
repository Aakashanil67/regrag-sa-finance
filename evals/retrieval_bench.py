"""Retrieval-only benchmark: how often each config/strategy puts the evidence in the top k.

    python -m evals.retrieval_bench --split dev --configs baseline,wordpiece,bge

Each config runs in its own subprocess because settings are read at import. No LLM is called.
The test split is refused until the pipeline is frozen.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from evals import configs
from evals.metrics import evidence_hit

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "reports" / "runs"
REPORT = ROOT / "reports" / "retrieval_bench.md"
SPLITS = {
    "dev": ROOT / "evals" / "questions_dev.jsonl",
    "test": ROOT / "evals" / "questions_test.jsonl",
}
STRATEGIES = ("semantic", "named_balanced", "bm25", "hybrid")


def first_hit_rank(item: dict, contexts: list[dict]) -> int | None:
    for rank, c in enumerate(contexts, start=1):
        if any(
            c["doc_id"] == e["doc_id"] and c["page_start"] <= e["page"] <= c["page_end"]
            for e in item["evidence"]
        ):
            return rank
    return None


def run_one(config: str, split: str, strategies: list[str]) -> dict:
    configs.apply(config)
    from src.chunking import _encode_tokens, get_embedding_tokenizer
    from src.config import RETRIEVAL_K, effective_settings
    from src.retrieve import retrieve
    from src.store import _get_model, get_collection

    items = [
        json.loads(line)
        for line in SPLITS[split].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items = [i for i in items if i["type"] != "unanswerable"]
    collection = get_collection()
    rows = []
    for strategy in strategies:
        for rerank in (False, True):
            anys = alls = multi_all = 0
            rr, latency = [], []
            for item in items:
                start = time.perf_counter()
                chunks = retrieve(
                    item["question"],
                    k=RETRIEVAL_K,
                    rerank=rerank,
                    strategy=strategy,
                    collection=collection,
                )
                latency.append((time.perf_counter() - start) * 1000)
                ctx = [
                    {"doc_id": c.doc_id, "page_start": c.page_start, "page_end": c.page_end}
                    for c in chunks
                ]
                a, b = evidence_hit(item, ctx)
                anys, alls = anys + a, alls + b
                multi_all += b and item["type"] == "multi"
                rank = first_hit_rank(item, ctx)
                rr.append(1 / rank if rank else 0.0)
            rows.append(
                {
                    "strategy": strategy,
                    "rerank": rerank,
                    "n": len(items),
                    "any_hit": anys,
                    "all_hit": alls,
                    "multi_all_hit": multi_all,
                    "multi_n": sum(1 for i in items if i["type"] == "multi"),
                    "mrr": round(statistics.mean(rr), 3),
                    "p50_ms": round(statistics.median(latency)),
                }
            )
    documents = collection.get(include=["documents"])["documents"]
    tokenizer = get_embedding_tokenizer()
    limit = _get_model().max_seq_length
    lengths = [len(_encode_tokens(d, tokenizer, add_special_tokens=True)) for d in documents]
    return {
        "config": config,
        "split": split,
        "settings": effective_settings(),
        "chunks": len(lengths),
        "embedding_limit": limit,
        "truncated_chunks": sum(n > limit for n in lengths),
        "rows": rows,
    }


def render(results: list[dict], split: str) -> str:
    lines = [
        f"Split `{split}`, answerable questions only, k=5. A hit means a retrieved chunk covers an evidence page.",
        "",
        "| config | chunks truncated at embedding | strategy | rerank | any hit | all docs hit | multi-doc all hit | MRR | p50 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        for row in r["rows"]:
            lines.append(
                f"| {r['config']} | {r['truncated_chunks']}/{r['chunks']} | {row['strategy']} | "
                f"{'yes' if row['rerank'] else 'no'} | {row['any_hit']}/{row['n']} | {row['all_hit']}/{row['n']} | "
                f"{row['multi_all_hit']}/{row['multi_n']} | {row['mrr']} | {row['p50_ms']} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=sorted(SPLITS), default="dev")
    ap.add_argument("--configs", default="baseline,wordpiece,bge")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--one")
    args = ap.parse_args(argv)
    strategies = args.strategies.split(",")
    if args.split == "test":
        protocol = json.loads((ROOT / "evals" / "protocol_test.json").read_text(encoding="utf-8"))
        if not protocol.get("frozen_pipeline"):
            sys.exit("the test split is only benchmarked after the pipeline is frozen")
    if args.one:
        print(json.dumps(run_one(args.one, args.split, strategies)))
        return
    results = []
    for name in args.configs.split(","):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "evals.retrieval_bench",
                "--one",
                name,
                "--split",
                args.split,
                "--strategies",
                args.strategies,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if args.split == "dev":
        for r in results:
            (RUNS_DIR / f"retrieval-{r['config']}.json").write_text(
                json.dumps(r, indent=1) + "\n", encoding="utf-8"
            )
    else:
        (RUNS_DIR / "retrieval-test.json").write_text(
            json.dumps(results, indent=1) + "\n", encoding="utf-8"
        )
    title = (
        "# Retrieval benchmark\n\n"
        if args.split == "dev"
        else "\n\nTest split, run once after the pipeline was frozen:\n\n"
    )
    mode = "w" if args.split == "dev" else "a"
    with REPORT.open(mode, encoding="utf-8") as fh:
        fh.write(title + render(results, args.split))
    print(render(results, args.split))


if __name__ == "__main__":
    main()
