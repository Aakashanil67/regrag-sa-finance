"""get_collection() must configure a wide-enough HNSW search width that identical queries against
an identical index return identical results across process launches — see src/config.py's
HNSW_SEARCH_EF comment for the non-determinism this fixes."""

from src import store
from src.config import HNSW_SEARCH_EF


def test_get_collection_sets_a_deterministic_search_width(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CHROMA_DIR", tmp_path)

    collection = store.get_collection()

    assert collection.metadata is not None
    assert collection.metadata.get("hnsw:search_ef") == HNSW_SEARCH_EF


def test_get_collection_repairs_an_existing_collection_missing_the_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CHROMA_DIR", tmp_path)
    store.get_collection().modify(metadata={"hnsw:search_ef": 10})

    collection = store.get_collection()

    assert collection.metadata.get("hnsw:search_ef") == HNSW_SEARCH_EF
