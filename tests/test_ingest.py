"""The three cleanup heuristics in ingest.py, tested directly against their inputs — no real PDF
needed, since each is a pure function over (page, text, font_size) tuples. One exception at the
bottom: a regression test against the real corpus PDF that surfaced a live extraction bug."""

import pytest

from src.config import CORPUS_DIR
from src.ingest import (
    _boilerplate_lines,
    _dehyphenate_join,
    _is_heading,
    _is_toc_page,
    extract_elements,
)


def test_boilerplate_line_repeated_across_most_pages_is_flagged():
    address_block = "P O Box 427 Pretoria 0001 South Africa"
    lines = [(page, address_block, 10.0) for page in range(1, 9)]  # every page, 8 total
    lines.append((1, "This sentence appears exactly once.", 10.0))

    boilerplate = _boilerplate_lines(lines, page_count=8)

    assert address_block in boilerplate
    assert "This sentence appears exactly once." not in boilerplate


def test_line_on_a_single_page_is_not_boilerplate_even_if_it_repeats_there():
    # a line repeated three times on the SAME page (e.g. a table's repeated label) touches only
    # one distinct page, so it must not be stripped as running header/footer noise
    lines = [(1, "Total", 10.0), (1, "Total", 10.0), (1, "Total", 10.0)]

    boilerplate = _boilerplate_lines(lines, page_count=10)

    assert "Total" not in boilerplate


def test_page_of_dot_leader_entries_is_detected_as_toc():
    toc_page = [
        ("1. Introduction ..................... 3", 10.0),
        ("2. Definitions ....................... 7", 10.0),
        ("3. Scope of application .............. 12", 10.0),
    ]

    assert _is_toc_page(toc_page) is True


def test_page_of_ordinary_prose_is_not_toc():
    prose_page = [
        ("This directive sets out the regulatory treatment of accounting provisions.", 10.0),
        ("Banks must classify impairments as either general or specific.", 10.0),
    ]

    assert _is_toc_page(prose_page) is False


def test_dehyphenate_join_merges_a_line_wrapped_word():
    lines = ["Banks issuing hy-", "brid debt instruments must comply."]

    assert _dehyphenate_join(lines) == "Banks issuing hybrid debt instruments must comply."


def test_dehyphenate_join_does_not_touch_a_real_hyphenated_word_on_one_line():
    assert (
        _dehyphenate_join(["A risk-based approach is required."])
        == "A risk-based approach is required."
    )


def test_numbered_heading_is_detected_regardless_of_font_size():
    assert _is_heading("4.2 Definitions", font_size=10.0, body_font_size=10.0) is True


def test_a_bare_date_is_not_mistaken_for_a_numbered_heading():
    # "08 July 2020" satisfies the same numbering pattern as "08 Introduction" would — a number,
    # whitespace, a word — and was silently dropped from a real chunk's text because of it (a
    # heading's own text never makes it into chunking.py's output, only its section metadata).
    assert _is_heading("08 July 2020", font_size=10.0, body_font_size=10.0) is False
    assert _is_heading("8 January 2024", font_size=10.0, body_font_size=10.0) is False


def test_a_sentence_ending_date_with_a_full_stop_is_not_a_heading_either():
    # same failure mode one document over: a comment deadline extracted as "28 February 2005."
    # (the sentence's own full stop still attached) slipped past the first version of this check,
    # which only matched a bare date with no trailing punctuation — found via a golden-set item
    # that refused because the deadline had been split off into a heading, leaving only "by not
    # later than" in the paragraph text.
    assert _is_heading("28 February 2005.", font_size=10.0, body_font_size=10.0) is False
    assert _is_heading("10 December 2004.", font_size=10.0, body_font_size=10.0) is False


def test_large_font_line_is_a_heading_even_without_numbering():
    assert _is_heading("Executive Summary", font_size=14.0, body_font_size=10.0) is True


def test_ordinary_sentence_at_body_size_is_not_a_heading():
    text = "This directive applies to all banks and controlling companies."
    assert _is_heading(text, font_size=10.0, body_font_size=10.0) is False


def test_press_release_dateline_survives_extraction_as_paragraph_text():
    # regression test against the real PDF that surfaced this bug, not just the unit-level regex:
    # confirms "08 July 2020" actually appears in a *paragraph* element's text after extraction,
    # not silently discarded as a heading's metadata-only text
    pdf_path = CORPUS_DIR / "fsca_press_conduct_standard_banks_2020.pdf"
    if not pdf_path.exists():
        pytest.skip("corpus not fetched — run python -m scripts.fetch_corpus first")

    elements = extract_elements(pdf_path)
    paragraph_text = " ".join(e.text for e in elements if e.kind == "paragraph")
    assert "08 July 2020" in paragraph_text


def test_comment_deadline_survives_extraction_as_paragraph_text():
    # same regression, second document: this circular's comment deadline was a sentence-ending
    # date ("...by not later than 28 February 2005.") rather than a standalone dateline, and it
    # took a golden-set eval failure (see reports/failure_analysis.md, item g08) to surface it
    pdf_path = CORPUS_DIR / "sarb_circular_19_2004_capital_hybrid_instruments.pdf"
    if not pdf_path.exists():
        pytest.skip("corpus not fetched — run python -m scripts.fetch_corpus first")

    elements = extract_elements(pdf_path)
    paragraph_text = " ".join(e.text for e in elements if e.kind == "paragraph")
    assert "28 February 2005" in paragraph_text
