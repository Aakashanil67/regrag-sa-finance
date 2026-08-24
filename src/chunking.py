"""Elements -> retrieval chunks.

Packs headings and paragraphs into ~500-token chunks (`CHUNK_TARGET_TOKENS`), preferring to break
at a section heading once a chunk already holds a reasonable amount of content, rather than
force-splitting on every heading — a document with fifty short numbered clauses would otherwise
produce fifty near-empty chunks. Consecutive chunks share a token-level overlap
(`CHUNK_OVERLAP_TOKENS`) so a sentence sitting right on a chunk boundary is still retrievable in
full from at least one of the two chunks. A paragraph bigger than the whole chunk budget on its
own (rare, but the NCA and some SARB directives have one) is split on sentence boundaries instead
of being force-fit or dropped.

Token counts use tiktoken's cl100k_base encoding as a fast, dependency-light proxy for chunk size
— not the tokenizer either Claude or the embedding model actually uses, but consistent and good
enough to budget against.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field

import tiktoken

from src.config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    CORPUS_DIR,
    MANIFEST_PATH,
    MIN_CHUNK_TOKENS,
    TOKENIZER_ENCODING,
)
from src.ingest import Element, extract_elements

_ENCODING = tiktoken.get_encoding(TOKENIZER_ENCODING)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING_FLUSH_FRACTION = 0.6  # flush the current chunk at a heading only once it's this full


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _overlap_tail(text: str, max_tokens: int) -> str:
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODING.decode(tokens[-max_tokens:])


@dataclass
class Chunk:
    doc_id: str
    index: int
    text: str
    page_start: int
    page_end: int
    section: str | None
    token_count: int = field(init=False)
    chunk_hash: str = field(init=False)

    def __post_init__(self):
        self.token_count = _token_count(self.text)
        # page_start, not just (doc_id, text): legislative documents genuinely repeat identical
        # passages at different pages (the NCA repeats definitional clauses across schedules) —
        # hashing on text alone would collapse two real locations into one stored vector and
        # silently keep whichever page happened to be ingested first, breaking that location's
        # citation. Not index: an edit earlier in the document would shift every later chunk's
        # index and defeat idempotent re-ingestion for content that didn't actually change.
        self.chunk_hash = hashlib.sha256(
            f"{self.doc_id}:{self.page_start}:{self.text}".encode()
        ).hexdigest()


def _hard_split_by_tokens(text: str) -> list[str]:
    """Last-resort token-window split for a single sentence that alone busts the chunk budget —
    legislative text (the NCA is the case that surfaced this) routinely runs a "provided that..."
    proviso for hundreds of words with no terminal punctuation, so sentence-level packing alone
    can't bound it. Not sentence-aware, so it can cut mid-clause; that's the accepted cost of
    keeping every chunk within roughly one embedding's worth of text."""
    tokens = _ENCODING.encode(text)
    step = max(CHUNK_TARGET_TOKENS - CHUNK_OVERLAP_TOKENS, 1)
    pieces = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_TARGET_TOKENS, len(tokens))
        pieces.append(_ENCODING.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += step
    return pieces


def _split_oversized_paragraph(
    text: str, page: int, section: str | None
) -> list[tuple[str, int, str | None]]:
    """Sentence-pack a too-big paragraph into <=CHUNK_TARGET_TOKENS pieces, same page/section."""
    sentences = _SENTENCE_SPLIT.split(text)
    pieces: list[tuple[str, int, str | None]] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _token_count(sentence)

        if sentence_tokens > CHUNK_TARGET_TOKENS:
            if current:
                pieces.append((" ".join(current), page, section))
                current, current_tokens = [], 0
            pieces.extend((piece, page, section) for piece in _hard_split_by_tokens(sentence))
            continue

        if current and current_tokens + sentence_tokens > CHUNK_TARGET_TOKENS:
            pieces.append((" ".join(current), page, section))
            overlap_text = _overlap_tail(" ".join(current), CHUNK_OVERLAP_TOKENS)
            current = [overlap_text] if overlap_text else []
            current_tokens = _token_count(overlap_text) if overlap_text else 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        pieces.append((" ".join(current), page, section))
    return pieces


def chunk_document(doc_id: str, elements: list[Element]) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    current_page_start: int | None = None
    current_page_end: int | None = None
    current_section: str | None = None

    def flush():
        nonlocal current_parts, current_tokens, current_page_start, current_page_end
        if not current_parts:
            return
        text = " ".join(current_parts)
        if chunks and chunks[-1].text == text:
            # current_parts is sometimes seeded verbatim from the previous chunk's own text (a
            # short chunk's "overlap tail" is the whole chunk, since _overlap_tail only trims
            # when there's something to trim) — if nothing new gets appended before the next
            # flush, that seed would otherwise re-emit as a byte-identical duplicate chunk.
            current_parts, current_tokens = [], 0
            return
        chunk = Chunk(
            doc_id=doc_id,
            index=len(chunks),
            text=text,
            page_start=current_page_start,
            page_end=current_page_end,
            section=current_section,
        )
        if chunk.token_count < MIN_CHUNK_TOKENS and chunks:
            previous = chunks[-1]
            merged_text = f"{previous.text} {text}"
            chunks[-1] = Chunk(
                doc_id=doc_id,
                index=previous.index,
                text=merged_text,
                page_start=previous.page_start,
                page_end=current_page_end,
                section=previous.section,
            )
        else:
            chunks.append(chunk)
        overlap_text = _overlap_tail(text, CHUNK_OVERLAP_TOKENS)
        current_parts = [overlap_text] if overlap_text else []
        current_tokens = _token_count(overlap_text) if overlap_text else 0
        current_page_start = current_page_end

    for element in elements:
        if element.kind == "heading":
            if current_tokens >= CHUNK_TARGET_TOKENS * _HEADING_FLUSH_FRACTION:
                flush()
            current_section = element.text
            continue

        paragraph_tokens = _token_count(element.text)

        if paragraph_tokens > CHUNK_TARGET_TOKENS:
            flush()
            # flush() just reseeded current_parts with an overlap tail meant for normal
            # packing continuity — irrelevant here, since the oversized paragraph's pieces are
            # appended directly below, bypassing current_parts entirely. Left in place, that
            # stale seed would surface as a spurious near-duplicate mini-chunk the next time
            # flush() runs (this bit two or more oversized paragraphs in a row in the NCA, whose
            # long legislative sentences trigger this path often). Discard it, then reseed from
            # the oversized paragraph's own last piece so overlap continuity carries forward
            # correctly into whatever comes next.
            current_parts = []
            current_tokens = 0

            last_piece_page = element.page
            for piece_text, piece_page, piece_section in _split_oversized_paragraph(
                element.text, element.page, current_section
            ):
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        index=len(chunks),
                        text=piece_text,
                        page_start=piece_page,
                        page_end=piece_page,
                        section=piece_section,
                    )
                )
                last_piece_page = piece_page

            overlap_text = _overlap_tail(chunks[-1].text, CHUNK_OVERLAP_TOKENS)
            current_parts = [overlap_text] if overlap_text else []
            current_tokens = _token_count(overlap_text) if overlap_text else 0
            current_page_start = last_piece_page
            current_page_end = last_piece_page
            continue

        if current_tokens + paragraph_tokens > CHUNK_TARGET_TOKENS and current_parts:
            flush()

        if current_page_start is None:
            current_page_start = element.page
        current_page_end = element.page
        current_parts.append(element.text)
        current_tokens += paragraph_tokens

    flush()
    return [c for c in chunks if c.text.strip()]


def chunk_corpus() -> dict[str, list[Chunk]]:
    """Ingest + chunk every document in the manifest. Keyed by doc_id for src/store.py."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    result = {}
    for entry in manifest:
        pdf_path = CORPUS_DIR / entry["filename"]
        if not pdf_path.exists():
            continue
        elements = extract_elements(pdf_path)
        result[entry["id"]] = chunk_document(entry["id"], elements)
    return result


if __name__ == "__main__":
    from src.report_chunks import write_chunk_quality_report

    write_chunk_quality_report(chunk_corpus())
