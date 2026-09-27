import json

import pytest

from src import store
from src.chunking import Chunk


class _FakeCollection:
    def __init__(self):
        self.rows = {}
        self.added_batches = []

    def count(self):
        return len(self.rows)

    def get(self, include=None):
        return {"ids": list(self.rows)}

    def add(self, *, ids, embeddings, documents, metadatas):
        self.added_batches.append((list(ids), list(embeddings)))
        self.rows.update(
            dict(zip(ids, zip(embeddings, documents, metadatas, strict=True), strict=True))
        )

    def delete(self, ids):
        for item_id in ids:
            self.rows.pop(item_id, None)


def _chunk(text="same text"):
    return Chunk(
        doc_id="doc-a",
        index=0,
        text=text,
        page_start=1,
        page_end=1,
        section="section",
    )


def test_index_change_reembeds_identical_chunk_ids_in_a_new_collection(tmp_path, monkeypatch):
    collections = {}
    monkeypatch.setattr(store, "CHROMA_DIR", tmp_path)
    monkeypatch.setattr(store, "chunk_corpus", lambda: {"doc-a": [_chunk()]})
    monkeypatch.setattr(
        store,
        "_get_collection_by_name",
        lambda name: collections.setdefault(name, _FakeCollection()),
    )
    fingerprints = iter(["index-a", "index-b"])
    monkeypatch.setattr("src.provenance.index_fingerprint", lambda: next(fingerprints))
    vectors = iter([[[1.0]], [[2.0]]])
    monkeypatch.setattr(store, "embed_texts", lambda texts: next(vectors))

    store.rebuild()
    first = store.rebuild()

    assert first["added"] == 1
    assert len(collections) == 2
    assert (
        list(collections.values())[0].added_batches[0][1]
        != list(collections.values())[1].added_batches[0][1]
    )


def test_failed_embedding_does_not_write_a_new_build_record(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CHROMA_DIR", tmp_path)
    (tmp_path / "build.json").write_text(
        json.dumps({"schema": 2, "index_fingerprint": "old", "chunk_count": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr(store, "chunk_corpus", lambda: {"doc-a": [_chunk()]})
    monkeypatch.setattr(store, "_get_collection_by_name", lambda name: _FakeCollection())
    monkeypatch.setattr("src.provenance.index_fingerprint", lambda: "new")

    def fail(texts):
        raise RuntimeError("synthetic embedding failure")

    monkeypatch.setattr(store, "embed_texts", fail)

    with pytest.raises(RuntimeError, match="synthetic embedding failure"):
        store.rebuild()

    assert (
        json.loads((tmp_path / "build.json").read_text(encoding="utf-8"))["index_fingerprint"]
        == "old"
    )
