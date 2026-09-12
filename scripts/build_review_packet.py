"""Exports a sealed holdout run into one standalone Markdown document a domain expert can audit
for entailment without running any code: for every item, the question, the generated answer, the
actual chunk text of every page it cited, what retrieval actually returned, and the reference
answer to compare against.

    python -m scripts.build_review_packet reports/runs/holdout-0cf5e0821ec7.json

Reuses evals.render_summary.load_run for the same artifact-schema check the summary renderer uses,
and src.retrieve.fetch_document_page for exact page lookups — deliberately not extended or
duplicated in src/retrieve.py itself, which evals/snapshot.py hashes for CI-fixture staleness and
has no reason to be touched just to export existing data.
"""

import argparse
import json
from pathlib import Path

from evals.render_summary import load_run
from src.config import GOLDEN_HOLDOUT_PATH, REPORTS_DIR
from src.retrieve import fetch_document_page
from src.store import get_collection

_HEADER = """# Holdout review packet

This is every item from a **sealed** holdout run (`evals/protocol.json`), exported so its
entailment — does the cited page actually support the claim being made — can be checked by someone
who knows this domain, without reading any code.

**This holdout is sealed.** `evals/protocol.json` forbids editing the holdout question/answer
files to make a result pass. If a reference answer below looks wrong, that is a real finding —
report it rather than correcting it in place. Fixing a genuine reference error requires a
documented protocol version bump and new sign-off, not a quiet edit.

For each item: the question, the generated answer, the actual text of every page it cited (so you
can check the citation supports the claim without opening the source PDF), what retrieval actually
returned (useful for a refused item — was the right material even available), and the reference
answer this project's author wrote when the holdout was built.
"""


def _dedupe_in_order(pairs):
    seen = set()
    out = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


class _ChunkTextCache:
    """Caches fetch_document_page per (doc_id, page) — cheap for a single call, but several
    holdout items cite the same heavily-chunked document (the National Credit Act alone has 333
    chunks), and this packet calls it once per citation across 30 items."""

    def __init__(self, collection):
        self._collection = collection
        self._cache: dict[tuple[str, int], list] = {}

    def get(self, doc_id: str, page: int):
        key = (doc_id, page)
        if key not in self._cache:
            self._cache[key] = fetch_document_page(doc_id, page, collection=self._collection)
        return self._cache[key]


def _retrieved_summary(chunk_ids: list[str], collection) -> list[str]:
    if not chunk_ids:
        return []
    result = collection.get(ids=chunk_ids, include=["metadatas"])
    pairs = _dedupe_in_order(
        (m["doc_id"], m["page_start"], m["page_end"]) for m in result["metadatas"]
    )
    return [
        f"{doc_id} p.{start}" if start == end else f"{doc_id} p.{start}-{end}"
        for doc_id, start, end in sorted(pairs)
    ]


def _render_item(
    item: dict, reference: dict, chunk_cache: _ChunkTextCache, collection
) -> list[str]:
    lines = [f"## {item['id']} ({item['type']}): {item['question']}", ""]

    lines.append("**Generated answer:**")
    lines.append("")
    lines.append(item["answer"])
    lines.append("")

    lines.append(f"**Reference answer:** {reference['reference_answer']}")
    lines.append("")

    cited_pages = _dedupe_in_order((c["doc_id"], c["page"]) for c in item["citations"])
    if cited_pages:
        lines.append("**Cited pages — actual chunk text:**")
        lines.append("")
        # a [doc, p.1-2] citation is stored page-expanded (one Citation row per page), and a chunk
        # spanning both pages resolves identically for each — dedupe by chunk_id so it prints once,
        # not once per cited page that happens to fall inside it
        seen_chunk_ids: set[str] = set()
        for doc_id, page in cited_pages:
            chunks = chunk_cache.get(doc_id, page)
            if not chunks:
                lines.append(
                    f"- `{doc_id}` p.{page}: **no chunk in the store covers this page** "
                    "— unverified citation"
                )
                continue
            for chunk in chunks:
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk.chunk_id)
                text = chunk.text.replace("\n", " ").strip()
                lines.append(f"- `{doc_id}` p.{chunk.page_start}-{chunk.page_end}:")
                lines.append(f"  - > {text}")
        lines.append("")

    retrieved = _retrieved_summary(item["retrieved_chunk_ids"], collection)
    lines.append(
        f"**Retrieved (top-k, all of it, not just what was cited):** {', '.join(retrieved) or 'nothing'}"
    )
    lines.append("")

    if "source_notices" in item:
        if item["source_notices"]:
            lines.append("**Source notices:**")
            for notice in item["source_notices"]:
                lines.append(f"- ({notice['kind']}) {notice['text']}")
        else:
            lines.append("**Source notices:** none")
    else:
        lines.append(
            "**Source notices:** not captured — this run predates source-notice recording "
            "(commit 45a1aad)"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def build_packet(run_path: Path, golden_path: Path = GOLDEN_HOLDOUT_PATH) -> str:
    run = load_run(run_path)
    reference_by_id = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in golden_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    collection = get_collection()
    chunk_cache = _ChunkTextCache(collection)

    lines = [
        _HEADER,
        f"**Run:** `{run.run_id}` | **Label:** `{run.label}` | **Status:** {run.status} | "
        f"**Items:** {len(run.items)}",
        "",
    ]
    for item in run.items:
        lines += _render_item(item, reference_by_id[item["id"]], chunk_cache, collection)

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="path to a reports/runs/<run_id>.json file")
    parser.add_argument(
        "--out", type=Path, default=REPORTS_DIR / "holdout_review_packet.md", help="output path"
    )
    args = parser.parse_args()

    packet = build_packet(args.artifact)
    args.out.write_text(packet, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
