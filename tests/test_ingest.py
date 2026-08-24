"""The three cleanup heuristics in ingest.py, tested directly against their inputs — no real PDF
needed, since each is a pure function over (page, text, font_size) tuples."""

from src.ingest import _boilerplate_lines, _dehyphenate_join, _is_heading, _is_toc_page


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


def test_large_font_line_is_a_heading_even_without_numbering():
    assert _is_heading("Executive Summary", font_size=14.0, body_font_size=10.0) is True


def test_ordinary_sentence_at_body_size_is_not_a_heading():
    text = "This directive applies to all banks and controlling companies."
    assert _is_heading(text, font_size=10.0, body_font_size=10.0) is False
