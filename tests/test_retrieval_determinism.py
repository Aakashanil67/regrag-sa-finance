"""The property src/retrieve.py's exact-cosine rewrite exists to guarantee: the same query, run in
two separate interpreters, returns the same top-k chunk ids. Nothing is mocked on purpose — a fake
collection cannot reproduce the original bug (ChromaDB's HNSW segment rebuilding its graph with a
CPU-count thread pool on each fresh process launch; see DECISIONS.md's "The v1.1 fixes"
section). Skipped, not failed, when the real vector store hasn't been built — it's gitignored and
neither CI nor a fresh clone has it before `python -m src.store --rebuild` runs.
"""

import json
import subprocess
import sys

import pytest

from src.config import CHROMA_DIR, ROOT

# Two queries, not one — a single query can be accidentally robust to the graph-construction
# non-determinism this guards against. Each query costs two full interpreter launches (embedding
# model load dominates, well over a minute here), so this stays the slowest test in the suite by
# a wide margin; that cost buys back a real, previously-shipped bug that no mocked test can catch.
_QUERIES = [
    "What date is Banks Act Circular 6/2004 dated?",
    "How many categories does the 2021 issued IFRS 9 text classify financial assets into?",
]

_SNIPPET = """
import json, sys
from src.retrieve import retrieve
results = retrieve(sys.argv[1], k=10, rerank=False)
print(json.dumps([c.chunk_id for c in results]))
"""


def _retrieve_in_a_fresh_process(query: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", _SNIPPET, query],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert (
        result.returncode == 0
    ), f"subprocess retrieval failed for query {query!r}:\n{result.stderr}"
    return json.loads(result.stdout)


def test_the_same_query_in_two_separate_interpreters_returns_the_same_top_k():
    if not (CHROMA_DIR / "chroma.sqlite3").exists():
        pytest.skip("vector store not built — run python -m src.store --rebuild first")

    for query in _QUERIES:
        first = _retrieve_in_a_fresh_process(query)
        second = _retrieve_in_a_fresh_process(query)
        assert first == second, f"retrieval for {query!r} differed across process launches"
        assert first, f"retrieval for {query!r} returned nothing — check the store is populated"
