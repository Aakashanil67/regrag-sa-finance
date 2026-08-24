"""PDF -> cleaned text elements.

PyMuPDF gives us each line's text plus its font size. That's enough to tell headings from body
text without an ML layout model: SARB/NCR/FSCA documents number their headings ("4.2 Definitions")
or set them in a larger font than the surrounding paragraph, almost always both. Neither signal
alone is reliable (numbered list items aren't headings; some documents use one font size
throughout), so `_is_heading` requires the numbering pattern OR a font size comfortably above the
document's own body-text size, never guesses from formatting alone, and accepts that a handful of
either false positive or false negative headings per document is the cost of not building a real
layout model for 19 PDFs.

Three cleanup passes run before chunking ever sees the text, because feeding chunking.py raw
PyMuPDF output would silently poison retrieval:
  1. repeated header/footer lines (address blocks, "Page X of Y") stripped by frequency across
     pages — real content doesn't repeat verbatim on 40%+ of a document's pages, boilerplate does.
  2. table-of-contents pages dropped by dot-leader density — a ToC entry retrieves as a false
     positive for whatever topic it's naming, without answering anything about it.
  3. de-hyphenated line wraps within a paragraph, so "hybrid-\ninstruments" reads as one word.
"""

import re
from dataclasses import dataclass

import fitz  # PyMuPDF

from src.config import HEADER_FOOTER_REPEAT_FRACTION, TOC_DOT_LEADER_FRACTION

_HEADING_NUMBERING = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,3})[\.\)]?\s+\S")
_DOT_LEADER = re.compile(r"\.{4,}\s*\d{1,4}\s*$")
_TRAILING_HYPHEN = re.compile(r"(\w)-$")


@dataclass
class Element:
    """One heading or paragraph, already cleaned, tagged with the page it came from."""

    kind: str  # "heading" | "paragraph"
    text: str
    page: int  # 1-indexed


def _line_font_size(line: dict) -> float:
    sizes = [span["size"] for span in line["spans"] if span["text"].strip()]
    return max(sizes) if sizes else 0.0


def _line_text(line: dict) -> str:
    return "".join(span["text"] for span in line["spans"]).strip()


def _extract_lines(doc: fitz.Document) -> list[tuple[int, str, float]]:
    """(page, text, font_size) for every non-empty line, reading order, across the whole doc."""
    lines = []
    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            for line in block.get("lines", []):
                text = _line_text(line)
                if text:
                    lines.append((page_index + 1, text, _line_font_size(line)))
    return lines


def _boilerplate_lines(lines: list[tuple[int, str, float]], page_count: int) -> set[str]:
    """Lines whose exact text repeats on enough distinct pages to be header/footer noise."""
    pages_seen: dict[str, set[int]] = {}
    for page, text, _ in lines:
        pages_seen.setdefault(text, set()).add(page)

    threshold = max(2, int(page_count * HEADER_FOOTER_REPEAT_FRACTION))
    return {text for text, pages in pages_seen.items() if len(pages) >= threshold}


def _is_toc_page(page_lines: list[tuple[str, float]]) -> bool:
    if not page_lines:
        return False
    dot_leader_count = sum(1 for text, _ in page_lines if _DOT_LEADER.search(text))
    return dot_leader_count / len(page_lines) >= TOC_DOT_LEADER_FRACTION


def _is_heading(text: str, font_size: float, body_font_size: float) -> bool:
    if len(text) > 120:  # headings are short; a numbered sentence-length line isn't one
        return False
    if _HEADING_NUMBERING.match(text):
        return True
    return font_size >= body_font_size * 1.15


def _dehyphenate_join(paragraph_lines: list[str]) -> str:
    joined = ""
    for line in paragraph_lines:
        if not joined:
            joined = line
        elif _TRAILING_HYPHEN.search(joined):
            joined = _TRAILING_HYPHEN.sub(r"\1", joined) + line
        else:
            joined = joined + " " + line
    return joined


def extract_elements(pdf_path) -> list[Element]:
    """Full pipeline: parse -> strip boilerplate -> drop ToC pages -> detect headings -> merge
    wrapped lines into paragraphs, de-hyphenated. Returns reading-order elements."""
    doc = fitz.open(pdf_path)
    try:
        lines = _extract_lines(doc)
        page_count = doc.page_count
    finally:
        doc.close()

    if not lines:
        return []

    boilerplate = _boilerplate_lines(lines, page_count)
    body_font_size = _median_font_size(lines)

    lines_by_page: dict[int, list[tuple[str, float]]] = {}
    for page, text, size in lines:
        if text in boilerplate:
            continue
        lines_by_page.setdefault(page, []).append((text, size))

    elements: list[Element] = []
    for page in sorted(lines_by_page):
        page_lines = lines_by_page[page]
        if _is_toc_page(page_lines):
            continue

        paragraph_buffer: list[str] = []
        for text, size in page_lines:
            if _is_heading(text, size, body_font_size):
                if paragraph_buffer:
                    elements.append(
                        Element(
                            kind="paragraph", text=_dehyphenate_join(paragraph_buffer), page=page
                        )
                    )
                    paragraph_buffer = []
                elements.append(Element(kind="heading", text=text, page=page))
            else:
                paragraph_buffer.append(text)
        if paragraph_buffer:
            elements.append(
                Element(kind="paragraph", text=_dehyphenate_join(paragraph_buffer), page=page)
            )

    return elements


def _median_font_size(lines: list[tuple[int, str, float]]) -> float:
    sizes = sorted(size for _, _, size in lines if size > 0)
    if not sizes:
        return 10.0
    return sizes[len(sizes) // 2]
