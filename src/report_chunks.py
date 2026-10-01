"""Chunk-quality report: size distribution and a garbage-chunk check.

Garbage here means a PDF-extraction artifact, not a subjective quality judgment: a chunk is
flagged if its token count falls under `MIN_CHUNK_TOKENS` (chunking.py's own merge step should
have absorbed these into a neighbour. Any that remain are documents with no real neighbour to
merge into) or if under half its characters are alphanumeric (a telltale sign of extracting a
table, a signature block, or garbled ligatures instead of prose).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.chunking import Chunk
from src.config import FIGURES_DIR, MIN_CHUNK_TOKENS, REPORTS_DIR

_ALNUM_RATIO_THRESHOLD = 0.5


def _is_garbage(chunk: Chunk) -> bool:
    if chunk.token_count < MIN_CHUNK_TOKENS:
        return True
    alnum = sum(1 for ch in chunk.text if ch.isalnum())
    return (alnum / len(chunk.text)) < _ALNUM_RATIO_THRESHOLD if chunk.text else True


def write_chunk_quality_report(chunks_by_doc: dict[str, list[Chunk]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    token_counts = [c.token_count for c in all_chunks]
    garbage = [c for c in all_chunks if _is_garbage(c)]
    empty = [c for c in all_chunks if not c.text.strip()]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(token_counts, bins=30, color="#2563eb", edgecolor="white")
    ax.set_xlabel("chunk size (tokens)")
    ax.set_ylabel("count")
    ax.set_title(
        f"Chunk size distribution — {len(all_chunks)} chunks across {len(chunks_by_doc)} documents"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "chunk_size_histogram.png", dpi=150)
    plt.close(fig)

    sorted_counts = sorted(token_counts)
    n = len(sorted_counts)
    median = sorted_counts[n // 2] if n else 0

    lines = [
        "# Chunk quality report",
        "",
        f"- Documents ingested: {len(chunks_by_doc)}",
        f"- Total chunks: {len(all_chunks)}",
        f"- Token count — min {min(token_counts, default=0)}, "
        f"median {median}, max {max(token_counts, default=0)}",
        f"- Empty chunks: {len(empty)}",
        f"- Garbage chunks (< {MIN_CHUNK_TOKENS} tokens or "
        f"< {_ALNUM_RATIO_THRESHOLD:.0%} alphanumeric): {len(garbage)} "
        f"({len(garbage) / len(all_chunks):.1%} of total)"
        if all_chunks
        else "- Garbage chunks: n/a (no chunks)",
        "",
        "![chunk size histogram](figures/chunk_size_histogram.png)",
        "",
        "## Per-document breakdown",
        "",
        "| doc_id | chunks | median tokens | min | max |",
        "|---|---|---|---|---|",
    ]
    for doc_id, chunks in sorted(chunks_by_doc.items()):
        counts = sorted(c.token_count for c in chunks)
        doc_median = counts[len(counts) // 2] if counts else 0
        lines.append(
            f"| {doc_id} | {len(chunks)} | {doc_median} | "
            f"{min(counts, default=0)} | {max(counts, default=0)} |"
        )

    if garbage:
        lines += ["", "## Garbage chunks flagged", ""]
        for chunk in garbage[:15]:
            preview = chunk.text[:120].replace("\n", " ")
            lines.append(
                f"- `{chunk.doc_id}` chunk {chunk.index} (page {chunk.page_start}): {preview!r}"
            )
        if len(garbage) > 15:
            lines.append(f"- ...and {len(garbage) - 15} more")

    (REPORTS_DIR / "chunk_quality.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"wrote reports/chunk_quality.md ({len(all_chunks)} chunks, {len(garbage)} flagged as garbage)"
    )
