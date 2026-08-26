"""retrieve.py's reranking logic, against mocked candidates and a mocked cross-encoder — no real
chromadb collection or embedding model needed to test the reordering behaviour itself."""

from src import retrieve as retrieve_module
from src.retrieve import RetrievedChunk, retrieve


def _chunk(chunk_id, text, score=0.5):
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        text=text,
        page_start=1,
        page_end=1,
        section=None,
        score=score,
    )


class _FakeCrossEncoder:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        return self._scores


def test_rerank_false_returns_bi_encoder_order_unchanged(monkeypatch):
    candidates = [_chunk("a", "first", 0.9), _chunk("b", "second", 0.8)]
    monkeypatch.setattr(
        retrieve_module, "_fetch_candidates", lambda q, n, d, collection=None: candidates
    )

    def fail_if_called():
        raise AssertionError("cross-encoder should not load when rerank=False")

    monkeypatch.setattr(retrieve_module, "_get_cross_encoder", fail_if_called)

    results = retrieve("a query", k=2, rerank=False)

    assert [c.chunk_id for c in results] == ["a", "b"]


def test_rerank_true_reorders_by_cross_encoder_score(monkeypatch):
    # bi-encoder ranks "a" first, but the cross-encoder disagrees and prefers "b"
    candidates = [_chunk("a", "first", 0.9), _chunk("b", "second", 0.8)]
    monkeypatch.setattr(
        retrieve_module, "_fetch_candidates", lambda q, n, d, collection=None: candidates
    )
    monkeypatch.setattr(
        retrieve_module, "_get_cross_encoder", lambda: _FakeCrossEncoder([1.0, 5.0])
    )

    results = retrieve("a query", k=2, rerank=True)

    assert [c.chunk_id for c in results] == ["b", "a"]
    assert results[0].score == 5.0


def test_rerank_true_requests_a_larger_candidate_pool(monkeypatch):
    monkeypatch.setattr(retrieve_module, "RERANK_CANDIDATE_POOL_SIZE", 20)
    requested_n = {}

    def fake_fetch(query, n, doc_ids, collection=None):
        requested_n["n"] = n
        return [_chunk("a", "only one candidate")]

    monkeypatch.setattr(retrieve_module, "_fetch_candidates", fake_fetch)
    monkeypatch.setattr(retrieve_module, "_get_cross_encoder", lambda: _FakeCrossEncoder([1.0]))

    retrieve("a query", k=5, rerank=True)

    assert requested_n["n"] == 20


def test_rerank_true_on_empty_candidates_returns_empty(monkeypatch):
    monkeypatch.setattr(retrieve_module, "_fetch_candidates", lambda q, n, d, collection=None: [])

    def fail_if_called():
        raise AssertionError("cross-encoder should not load when there's nothing to rerank")

    monkeypatch.setattr(retrieve_module, "_get_cross_encoder", fail_if_called)

    assert retrieve("a query", k=5, rerank=True) == []
