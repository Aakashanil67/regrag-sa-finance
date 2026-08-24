"""Edge cases in chunking.py: the parts that are easy to get subtly wrong, not the happy path."""

from src import chunking
from src.chunking import chunk_document
from src.ingest import Element


def _words(n: int, word: str = "regulation") -> str:
    """n copies of a common word, space-separated — under cl100k_base each is one token, so this
    gives an approximately-known token count without depending on tiktoken internals directly."""
    return " ".join([word] * n)


def test_heading_with_no_body_produces_no_chunk():
    elements = [Element(kind="heading", text="1. Introduction", page=1)]

    chunks = chunk_document("doc1", elements)

    assert chunks == []


def test_oversized_paragraph_splits_under_the_chunk_budget(monkeypatch):
    monkeypatch.setattr(chunking, "CHUNK_TARGET_TOKENS", 20)
    monkeypatch.setattr(chunking, "CHUNK_OVERLAP_TOKENS", 5)

    # one paragraph, no sentence breaks near the token budget — five long "sentences"
    text = ". ".join([_words(15) for _ in range(5)]) + "."
    elements = [Element(kind="paragraph", text=text, page=3)]

    chunks = chunk_document("doc1", elements)

    assert len(chunks) > 1
    # a little slack: the overlap carryover from the previous piece plus one more sentence can
    # push a piece slightly past the budget — it must never approach the full unsplit paragraph
    for chunk in chunks:
        assert chunk.token_count < chunking.CHUNK_TARGET_TOKENS * 2


def test_consecutive_chunks_share_overlap_text(monkeypatch):
    monkeypatch.setattr(chunking, "CHUNK_TARGET_TOKENS", 30)
    monkeypatch.setattr(chunking, "CHUNK_OVERLAP_TOKENS", 10)
    monkeypatch.setattr(chunking, "MIN_CHUNK_TOKENS", 2)

    elements = [
        Element(kind="paragraph", text=_words(28), page=1),
        Element(kind="paragraph", text=_words(28, word="threshold"), page=1),
    ]

    chunks = chunk_document("doc1", elements)

    assert len(chunks) == 2
    tail_of_first = chunks[0].text.split()[-5:]
    start_of_second = chunks[1].text.split()[:5]
    assert tail_of_first == start_of_second


def test_chunk_spanning_pages_records_start_and_end_page(monkeypatch):
    monkeypatch.setattr(chunking, "CHUNK_TARGET_TOKENS", 100)
    monkeypatch.setattr(chunking, "CHUNK_OVERLAP_TOKENS", 10)

    elements = [
        Element(kind="paragraph", text=_words(10), page=4),
        Element(kind="paragraph", text=_words(10), page=5),
        Element(kind="paragraph", text=_words(10), page=6),
    ]

    chunks = chunk_document("doc1", elements)

    assert len(chunks) == 1
    assert chunks[0].page_start == 4
    assert chunks[0].page_end == 6
