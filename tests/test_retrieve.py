"""retrieve.py's reranking logic, against mocked candidates and a mocked cross-encoder — no real
chromadb collection or embedding model needed to test the reordering behaviour itself."""

from src import retrieve as retrieve_module
from src.retrieve import RetrievedChunk, _fetch_candidates, fetch_document_page, retrieve


class _FakeCollection:
    """Minimal stand-in for chromadb's collection.get(), keyed on the where clause's doc_id."""

    def __init__(self, ids, documents, metadatas, embeddings=None):
        self._ids = ids
        self._documents = documents
        self._metadatas = metadatas
        self._embeddings = embeddings

    def get(self, where=None, include=None):
        if where and "doc_id" in where and isinstance(where["doc_id"], str):
            keep = [i for i, m in enumerate(self._metadatas) if m["doc_id"] == where["doc_id"]]
        elif where and "doc_id" in where:
            wanted = set(where["doc_id"]["$in"])
            keep = [i for i, m in enumerate(self._metadatas) if m["doc_id"] in wanted]
        else:
            keep = list(range(len(self._ids)))
        result = {
            "ids": [self._ids[i] for i in keep],
            "documents": [self._documents[i] for i in keep],
            "metadatas": [self._metadatas[i] for i in keep],
        }
        if self._embeddings is not None:
            result["embeddings"] = [self._embeddings[i] for i in keep]
        return result


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


def test_fetch_candidates_ranks_by_exact_cosine_similarity(monkeypatch):
    # "b" points the same direction as the query; "a" is orthogonal; "c" points opposite —
    # cosine similarity must rank b > a > c regardless of each vector's raw magnitude.
    collection = _FakeCollection(
        ids=["a", "b", "c"],
        documents=["orthogonal", "aligned", "opposite"],
        metadatas=[
            {"doc_id": "doc1", "page_start": 1, "page_end": 1, "section": ""},
            {"doc_id": "doc1", "page_start": 1, "page_end": 1, "section": ""},
            {"doc_id": "doc1", "page_start": 1, "page_end": 1, "section": ""},
        ],
        embeddings=[[0.0, 5.0], [3.0, 0.0], [-1.0, 0.0]],
    )
    monkeypatch.setattr(retrieve_module, "embed_texts", lambda texts: [[1.0, 0.0]])

    results = _fetch_candidates("query", n=3, doc_ids=None, collection=collection)

    assert [c.chunk_id for c in results] == ["b", "a", "c"]


def test_fetch_candidates_is_exact_not_approximate(monkeypatch):
    # a larger candidate set than any plausible approximate search width — exact search must
    # still find the single best match, unlike an ANN index tuned for a smaller corpus
    embeddings = [[0.0, 1.0]] * 50 + [[1.0, 0.0]]
    collection = _FakeCollection(
        ids=[f"filler{i}" for i in range(50)] + ["best"],
        documents=["filler"] * 50 + ["best match"],
        metadatas=[{"doc_id": "doc1", "page_start": 1, "page_end": 1, "section": ""}] * 51,
        embeddings=embeddings,
    )
    monkeypatch.setattr(retrieve_module, "embed_texts", lambda texts: [[1.0, 0.0]])

    results = _fetch_candidates("query", n=1, doc_ids=None, collection=collection)

    assert [c.chunk_id for c in results] == ["best"]


def test_fetch_candidates_filters_by_doc_ids(monkeypatch):
    collection = _FakeCollection(
        ids=["a", "b"],
        documents=["in scope", "out of scope"],
        metadatas=[
            {"doc_id": "doc1", "page_start": 1, "page_end": 1, "section": ""},
            {"doc_id": "doc2", "page_start": 1, "page_end": 1, "section": ""},
        ],
        embeddings=[[1.0, 0.0], [1.0, 0.0]],
    )
    monkeypatch.setattr(retrieve_module, "embed_texts", lambda texts: [[1.0, 0.0]])

    results = _fetch_candidates("query", n=5, doc_ids=["doc1"], collection=collection)

    assert [c.chunk_id for c in results] == ["a"]


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
