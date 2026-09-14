"""Tokenizer-aware chunk budgets and the metadata/provenance they produce."""

import pytest

from scripts import audit_token_budget
from src import chunking, provenance, store
from src.chunking import Chunk, chunk_document
from src.ingest import Element


class FakeWordpieceTokenizer:
    """Small deterministic stand-in for a WordPiece tokenizer.

    Each non-whitespace token is one body token. The two wrapper tokens model the
    special-token overhead used by the embedding model.
    """

    name_or_path = "synthetic-wordpiece-v1"
    cls_token_id = 101
    sep_token_id = 102

    def __init__(self):
        self._token_to_id = {}
        self._id_to_token = {}

    def encode(self, text, add_special_tokens=True, truncation=False):
        del truncation
        body = []
        for token in text.split():
            token_id = self._token_to_id.setdefault(token, 1000 + len(self._token_to_id))
            self._id_to_token[token_id] = token
            body.append(token_id)
        return (
            ([self.cls_token_id] if add_special_tokens else [])
            + body
            + ([self.sep_token_id] if add_special_tokens else [])
        )

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        body = [
            token_id
            for token_id in token_ids
            if token_id not in {self.cls_token_id, self.sep_token_id}
        ]
        return " ".join(self._id_to_token[token_id] for token_id in body)


def test_tokenizer_budget_includes_special_tokens_and_legacy_count_remains_available():
    tokenizer = FakeWordpieceTokenizer()
    chunks = chunk_document(
        "doc-a",
        [Element(kind="paragraph", text="alpha beta gamma delta epsilon", page=4)],
        tokenizer=tokenizer,
        target_tokens=5,
        overlap_tokens=1,
    )

    assert chunks
    assert all(chunk.embedding_token_count <= 5 for chunk in chunks)
    assert all(chunk.token_count == chunking._token_count(chunk.text) for chunk in chunks)
    assert chunks[0].embedding_token_count == 5  # three body tokens plus two specials


def test_tokenizer_chunking_preserves_unicode_tail_and_page_metadata():
    tokenizer = FakeWordpieceTokenizer()
    elements = [
        Element(kind="heading", text="§ 69(2) α", page=1),
        Element(kind="paragraph", text="one two three four five six", page=1),
        Element(kind="paragraph", text="seven eight nine ten", page=2),
    ]

    chunks = chunk_document(
        "doc-a", elements, tokenizer=tokenizer, target_tokens=7, overlap_tokens=3
    )

    assert len(chunks) >= 2
    assert chunks[0].section == "§ 69(2) α"
    assert chunks[-1].page_end == 2
    assert all(chunk.embedding_token_count <= 7 for chunk in chunks)
    assert set(chunks[0].text.split()[-2:]) & set(chunks[1].text.split()[:3])


def test_tokenizer_hard_split_keeps_long_legislative_sentence_within_budget():
    tokenizer = FakeWordpieceTokenizer()
    sentence = "provided that " + "the bank must comply " * 20

    chunks = chunk_document(
        "doc-a",
        [Element(kind="paragraph", text=sentence, page=3)],
        tokenizer=tokenizer,
        target_tokens=9,
        overlap_tokens=2,
    )

    assert len(chunks) > 1
    assert all(chunk.embedding_token_count <= 9 for chunk in chunks)
    assert "provided" in chunks[0].text
    assert "comply" in chunks[-1].text


def test_tokenizer_overlap_must_be_smaller_than_budget():
    with pytest.raises(ValueError, match="overlap.*budget"):
        chunk_document(
            "doc-a",
            [Element(kind="paragraph", text="one two", page=1)],
            tokenizer=FakeWordpieceTokenizer(),
            target_tokens=8,
            overlap_tokens=8,
        )


def test_store_metadata_records_legacy_and_embedding_token_counts():
    chunk = Chunk("doc-a", 0, "same text", 1, 1, "section", embedding_token_count=7)

    metadata = store._chunk_metadata(chunk)

    assert metadata["token_count"] == chunk.token_count
    assert metadata["embedding_token_count"] == 7


def test_index_fingerprint_changes_when_tokenizer_budget_changes(monkeypatch):
    first = provenance.index_fingerprint()
    monkeypatch.setattr(provenance, "TOKENIZER_AWARE_CHUNK_TARGET_TOKENS", 241)
    second = provenance.index_fingerprint()

    assert first != second


def test_token_budget_audit_rejects_non_development_splits():
    with pytest.raises(ValueError, match="development split"):
        audit_token_budget.validate_split("holdout")


def test_token_budget_percentiles_are_deterministic():
    assert audit_token_budget.percentiles([1, 2, 3, 4, 5]) == {
        "count": 5,
        "p50": 3.0,
        "p95": 4.8,
        "max": 5,
    }
