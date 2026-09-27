import pytest

from src import retrieve
from src.retrieve import RetrievedChunk


class FakeCollection:
    def __init__(self, docs, name="fake"):
        self.docs, self.name = docs, name

    def count(self):
        return len(self.docs)

    def get(self, where=None, include=None):
        return {
            "ids": [d[0] for d in self.docs],
            "documents": [d[1] for d in self.docs],
            "metadatas": [
                {"doc_id": d[2], "page_start": 1, "page_end": 1, "section": ""} for d in self.docs
            ],
        }


DOCS = [
    ("a", "Directive 8/2025 sets the threshold amounts for the revised approaches", "d8_2025"),
    ("b", "threshold amounts and credit risk in general terms", "guide"),
    ("c", "principles for operational resilience", "d4_2023"),
]


@pytest.fixture(autouse=True)
def _clear_cache():
    retrieve._bm25_cache.clear()


def _c(chunk_id):
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id=chunk_id, text="", page_start=1, page_end=1, section="", score=0.0
    )


def test_bm25_ranks_the_exact_identifier_first():
    out = retrieve._bm25_candidates("What does Directive 8/2025 set?", 3, FakeCollection(DOCS))
    assert out[0].chunk_id == "a"


def test_bm25_tokens_keep_instrument_numbers_whole():
    assert "8/2025" in retrieve._bm25_tokens("Directive 8/2025, s 103(5)")


def test_rrf_rewards_items_both_lists_agree_on():
    fused = retrieve._rrf([[_c("x"), _c("y"), _c("z")], [_c("y")]], 3)
    assert [c.chunk_id for c in fused] == ["y", "x", "z"]


def test_hybrid_is_deterministic(monkeypatch):
    monkeypatch.setattr(
        retrieve, "_fetch_candidates", lambda q, n, doc_ids, collection=None: [_c("c"), _c("b")]
    )
    col = FakeCollection(DOCS)
    first = retrieve.retrieve("Directive 8/2025 thresholds", k=3, strategy="hybrid", collection=col)
    second = retrieve.retrieve(
        "Directive 8/2025 thresholds", k=3, strategy="hybrid", collection=col
    )
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunk_id in {"a", "b"}


def test_bm25_index_rebuilds_when_the_collection_changes():
    retrieve._bm25_candidates("threshold", 1, FakeCollection(DOCS))
    out = retrieve._bm25_candidates("resilience", 1, FakeCollection(DOCS[2:]))
    assert out[0].chunk_id == "c"
