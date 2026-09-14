"""Elements -> retrieval chunks.

Packs headings and paragraphs into ~800-token chunks (`CHUNK_TARGET_TOKENS`), preferring to break
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
    EMBEDDING_TOKENIZER_NAME,
    MANIFEST_PATH,
    MIN_CHUNK_TOKENS,
    TOKENIZER_AWARE_CHUNK_OVERLAP_TOKENS,
    TOKENIZER_AWARE_CHUNK_TARGET_TOKENS,
    TOKENIZER_ENCODING,
)
from src.ingest import Element, extract_elements

_ENCODING = tiktoken.get_encoding(TOKENIZER_ENCODING)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING_FLUSH_FRACTION = 0.6  # flush the current chunk at a heading only once it's this full
_EMBEDDING_TOKENIZER = None


def get_embedding_tokenizer():
    """Load the embedding tokenizer only when a tokenizer-aware path requests it."""
    global _EMBEDDING_TOKENIZER
    if _EMBEDDING_TOKENIZER is None:
        from transformers import AutoTokenizer

        _EMBEDDING_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_TOKENIZER_NAME)
    return _EMBEDDING_TOKENIZER


def _encode_tokens(text: str, tokenizer=None, *, add_special_tokens: bool = False) -> list[int]:
    if tokenizer is None:
        return _ENCODING.encode(text)
    return list(
        tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )
    )


def _special_token_count(tokenizer) -> int:
    return len(_encode_tokens("", tokenizer, add_special_tokens=True))


def _token_count(text: str, tokenizer=None) -> int:
    """Return legacy tiktoken counts, or special-token-inclusive model counts."""
    if tokenizer is None:
        return len(_ENCODING.encode(text))
    return len(_encode_tokens(text, tokenizer, add_special_tokens=True))


def _packing_token_count(text: str, tokenizer=None) -> int:
    """Count content tokens for packing; a tokenizer-aware chunk adds specials once at flush."""
    return len(_encode_tokens(text, tokenizer, add_special_tokens=False))


def _overlap_tail(text: str, max_tokens: int, tokenizer=None) -> str:
    tokens = _encode_tokens(text, tokenizer, add_special_tokens=False)
    if tokenizer is None:
        if len(tokens) <= max_tokens:
            return text
        return _ENCODING.decode(tokens[-max_tokens:])

    body_limit = max(max_tokens - _special_token_count(tokenizer), 0)
    if len(tokens) <= body_limit:
        return text
    if not body_limit:
        return ""
    return tokenizer.decode(tokens[-body_limit:], skip_special_tokens=True)


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
    embedding_token_count: int | None = None
    embedding_tokenizer: str | None = None

    def __post_init__(self):
        self.token_count = _token_count(self.text)
        # Location metadata is part of the identity: legislative documents genuinely repeat
        # identical passages at different pages, and a chunk can span a different page range or
        # section after extraction changes. Hashing only text/page_start would silently retain
        # stale citation metadata in the vector store. Not index: an edit earlier in the document
        # would shift every later chunk index and defeat idempotent re-ingestion unnecessarily.
        self.chunk_hash = hashlib.sha256(
            f"{self.doc_id}:{self.page_start}:{self.page_end}:{self.section or ''}:{self.text}".encode()
        ).hexdigest()


def _hard_split_by_tokens(
    text: str, *, tokenizer=None, target_tokens: int, overlap_tokens: int
) -> list[str]:
    """Last-resort token-window split for a single sentence that alone busts the chunk budget —
    legislative text (the NCA is the case that surfaced this) routinely runs a "provided that..."
    proviso for hundreds of words with no terminal punctuation, so sentence-level packing alone
    can't bound it. Not sentence-aware, so it can cut mid-clause; that's the accepted cost of
    keeping every chunk within roughly one embedding's worth of text."""
    tokens = _encode_tokens(text, tokenizer, add_special_tokens=False)
    special_tokens = _special_token_count(tokenizer) if tokenizer is not None else 0
    body_budget = target_tokens - special_tokens
    body_overlap = max(overlap_tokens - special_tokens, 0)
    step = max(body_budget - body_overlap, 1)
    pieces = []
    start = 0
    while start < len(tokens):
        end = min(start + body_budget, len(tokens))
        if tokenizer is None:
            pieces.append(_ENCODING.decode(tokens[start:end]))
        else:
            pieces.append(tokenizer.decode(tokens[start:end], skip_special_tokens=True))
        if end == len(tokens):
            break
        start += step
    return pieces


def _split_oversized_paragraph(
    text: str,
    page: int,
    section: str | None,
    *,
    tokenizer=None,
    target_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, int, str | None]]:
    """Sentence-pack a too-big paragraph into <=CHUNK_TARGET_TOKENS pieces, same page/section."""
    sentences = _SENTENCE_SPLIT.split(text)
    pieces: list[tuple[str, int, str | None]] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _packing_token_count(sentence, tokenizer)

        special_tokens = _special_token_count(tokenizer) if tokenizer is not None else 0
        body_budget = target_tokens - special_tokens
        if sentence_tokens > body_budget:
            if current:
                pieces.append((" ".join(current), page, section))
                current, current_tokens = [], 0
            pieces.extend(
                (piece, page, section)
                for piece in _hard_split_by_tokens(
                    sentence,
                    tokenizer=tokenizer,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
            continue

        if current and current_tokens + sentence_tokens > body_budget:
            pieces.append((" ".join(current), page, section))
            overlap_text = _overlap_tail(" ".join(current), overlap_tokens, tokenizer)
            current = [overlap_text] if overlap_text else []
            current_tokens = _packing_token_count(overlap_text, tokenizer) if overlap_text else 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        pieces.append((" ".join(current), page, section))
    return pieces


def chunk_document(
    doc_id: str,
    elements: list[Element],
    *,
    tokenizer=None,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    if tokenizer is None:
        target_tokens = CHUNK_TARGET_TOKENS if target_tokens is None else target_tokens
        overlap_tokens = CHUNK_OVERLAP_TOKENS if overlap_tokens is None else overlap_tokens
    else:
        target_tokens = (
            TOKENIZER_AWARE_CHUNK_TARGET_TOKENS if target_tokens is None else target_tokens
        )
        overlap_tokens = (
            TOKENIZER_AWARE_CHUNK_OVERLAP_TOKENS if overlap_tokens is None else overlap_tokens
        )
    if target_tokens <= 0:
        raise ValueError("target token budget must be positive")
    if overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("overlap token budget must be smaller than the chunk token budget")

    special_tokens = _special_token_count(tokenizer) if tokenizer is not None else 0
    body_budget = target_tokens - special_tokens
    if tokenizer is not None and body_budget <= 0:
        raise ValueError("chunk token budget must exceed tokenizer special-token overhead")

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    current_page_start: int | None = None
    current_page_end: int | None = None
    current_section: str | None = None

    def packed_count(text: str) -> int:
        return _packing_token_count(text, tokenizer)

    def make_chunk(text: str, page_start: int, page_end: int, section: str | None) -> Chunk:
        return Chunk(
            doc_id=doc_id,
            index=len(chunks),
            text=text,
            page_start=page_start,
            page_end=page_end,
            section=section,
            embedding_token_count=_token_count(text, tokenizer) if tokenizer is not None else None,
            embedding_tokenizer=(
                getattr(tokenizer, "name_or_path", None) if tokenizer is not None else None
            ),
        )

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
        chunk = make_chunk(text, current_page_start, current_page_end, current_section)
        merged_text = f"{chunks[-1].text} {text}" if chunks else text
        can_merge_without_exceeding_budget = (
            tokenizer is None or _token_count(merged_text, tokenizer) <= target_tokens
        )
        if chunk.token_count < MIN_CHUNK_TOKENS and chunks and can_merge_without_exceeding_budget:
            previous = chunks[-1]
            chunks[-1] = make_chunk(
                merged_text, previous.page_start, current_page_end, previous.section
            )
        else:
            chunks.append(chunk)
        overlap_text = _overlap_tail(text, overlap_tokens, tokenizer)
        current_parts = [overlap_text] if overlap_text else []
        current_tokens = packed_count(overlap_text) if overlap_text else 0
        current_page_start = current_page_end

    for element in elements:
        if element.kind == "heading":
            if current_tokens >= CHUNK_TARGET_TOKENS * _HEADING_FLUSH_FRACTION:
                flush()
            current_section = element.text
            continue

        paragraph_tokens = packed_count(element.text)

        if paragraph_tokens > body_budget:
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
                element.text,
                element.page,
                current_section,
                tokenizer=tokenizer,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            ):
                chunks.append(make_chunk(piece_text, piece_page, piece_page, piece_section))
                last_piece_page = piece_page

            overlap_text = _overlap_tail(chunks[-1].text, overlap_tokens, tokenizer)
            current_parts = [overlap_text] if overlap_text else []
            current_tokens = packed_count(overlap_text) if overlap_text else 0
            current_page_start = last_piece_page
            current_page_end = last_piece_page
            continue

        if current_tokens + paragraph_tokens > body_budget and current_parts:
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
