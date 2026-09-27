import src.ingest as ingest

_PAGES = [
    "1. Purpose\n\nBanks must treat customers fairly.",
    "2. Scope\n\nThis applies to all banks.",
]


class _StubDoc:
    page_count = 2

    def close(self):
        pass


def test_ocr_elements_split_headings_and_paragraphs_by_page(monkeypatch):
    monkeypatch.setattr(ingest.fitz, "open", lambda _path: _StubDoc())
    monkeypatch.setattr(ingest, "ocr_page_text", lambda _doc, i, _doc_id: _PAGES[i])

    elements = ingest.extract_ocr_elements("ignored.pdf", "doc")

    assert [(e.kind, e.text, e.page) for e in elements] == [
        ("heading", "1. Purpose", 1),
        ("paragraph", "Banks must treat customers fairly.", 1),
        ("heading", "2. Scope", 2),
        ("paragraph", "This applies to all banks.", 2),
    ]
