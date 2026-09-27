"""Build a packet and blank form for a human review of served answers.

python -m evals.review_packet reports/runs/test-final.json --n 20 --seed 7
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def select(items: list[dict], n: int, seed: int) -> list[dict]:
    pool = [i for i in items if i["type"] != "unanswerable" and not i["refused"]]
    pool.sort(key=lambda i: hashlib.sha256(f"{seed}:{i['id']}".encode()).hexdigest())
    return pool[:n]


def cited_pages(item: dict) -> list[tuple[str, int, str]]:
    """Text of every saved context covering a cited page, once per chunk."""
    seen, out = set(), []
    for cite in item["citations"]:
        for ctx in item["contexts"]:
            covers = ctx["doc_id"] == cite["doc_id"] and (
                ctx["page_start"] <= cite["page"] <= ctx["page_end"]
            )
            if covers and ctx["chunk_id"] not in seen:
                seen.add(ctx["chunk_id"])
                out.append((ctx["doc_id"], cite["page"], ctx["text"].replace("\u00a0", " ")))
    return out


def render(items: list[dict]) -> str:
    lines = [
        "# Review packet",
        "",
        "For each answer, check it against the quoted source text and your own knowledge.",
        "Label correctness (against the reference answer) and support (against the cited text)",
        "in reports/review_v1.2.csv. Definitions are in reports/reviewer_instructions.md.",
        "",
    ]
    for n, item in enumerate(items, 1):
        lines += [f"### {n}. {item['id']}", "", f"**Question:** {item['question']}", ""]
        lines += [f"**Answer:** {item['served_answer']}", "", "**Cited text:**", ""]
        for doc_id, page, text in cited_pages(item):
            lines += [f"> [{doc_id}, p.{page}] {text}", ""]
        lines += [f"**Reference answer:** {item['reference_answer']}", "", "**Evidence:**", ""]
        for ev in item["evidence"]:
            lines.append(f"- [{ev['doc_id']}, p.{ev['page']}] \"{ev['quote']}\"")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="reports/review_packet_v1.2.md")
    p.add_argument("--form", default="reports/review_v1.2.csv")
    args = p.parse_args()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    items = select(run["items"], args.n, args.seed)
    Path(args.out).write_text(render(items), encoding="utf-8")
    with open(args.form, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "correctness", "support", "reviewer", "notes"])
        for item in items:
            w.writerow([item["id"], "", "", "", ""])


if __name__ == "__main__":
    main()
