"""retrieve.py's reranking logic, against mocked candidates and a mocked cross-encoder — no real
chromadb collection or embedding model needed to test the reordering behaviour itself."""

from src import retrieve as retrieve_module
from src.retrieve import RetrievedChunk, fetch_document_page, retrieve


class _FakeCollection:
    """Minimal stand-in for chromadb's collection.get(), keyed on the where clause's doc_id."""

    def __init__(self, ids, documents, metadatas):
        self._ids = ids
        self._documents = documents
        self._metadatas = metadatas

    def get(self, where=None, include=None):
        doc_id = where["doc_id"]
        keep = [i for i, m in enumerate(self._metadatas) if m["doc_id"] == doc_id]
        return {
            "ids": [self._ids[i] for i in keep],
            "documents": [self._documents[i] for i in keep],
            "metadatas": [self._metadatas[i] for i in keep],
        }


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


def test_fetch_document_page_returns_only_chunks_covering_that_page():
    collection = _FakeCollection(
        ids=["c1", "c2", "c3"],
        documents=["page 1-2 text", "page 3-5 text", "other doc"],
        metadatas=[
            {"doc_id": "doc_a", "page_start": 1, "page_end": 2, "section": ""},
            {"doc_id": "doc_a", "page_start": 3, "page_end": 5, "section": ""},
            {"doc_id": "doc_b", "page_start": 1, "page_end": 1, "section": ""},
        ],
    )

    result = fetch_document_page("doc_a", page=4, collection=collection)

    assert [c.chunk_id for c in result] == ["c2"]


def test_fetch_document_page_cannot_regress_to_last_chunk_wins():
    # several chunks from the same document, requested page only covered by an earlier one —
    # this exact bug once made a citation check keep only the last chunk's page range
    collection = _FakeCollection(
        ids=["c1", "c2", "c3"],
        documents=["first", "second", "third"],
        metadatas=[
            {"doc_id": "doc_a", "page_start": 1, "page_end": 1, "section": ""},
            {"doc_id": "doc_a", "page_start": 2, "page_end": 2, "section": ""},
            {"doc_id": "doc_a", "page_start": 3, "page_end": 3, "section": ""},
        ],
    )

    result = fetch_document_page("doc_a", page=1, collection=collection)

    assert [c.chunk_id for c in result] == ["c1"]


def test_fetch_document_page_returns_empty_for_an_uncovered_page():
    collection = _FakeCollection(
        ids=["c1"],
        documents=["text"],
        metadatas=[{"doc_id": "doc_a", "page_start": 1, "page_end": 1, "section": ""}],
    )

    assert fetch_document_page("doc_a", page=99, collection=collection) == []


def test_fetch_document_page_sorts_by_page_then_chunk_id():
    collection = _FakeCollection(
        ids=["z", "a"],
        documents=["later chunk", "earlier chunk"],
        metadatas=[
            {"doc_id": "doc_a", "page_start": 1, "page_end": 3, "section": ""},
            {"doc_id": "doc_a", "page_start": 1, "page_end": 1, "section": ""},
        ],
    )

    result = fetch_document_page("doc_a", page=1, collection=collection)

    assert [c.chunk_id for c in result] == ["a", "z"]
