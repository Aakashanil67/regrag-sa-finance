from evals.build_explorer import build_data, cited_passages, render


def test_cited_passages_only_covers_cited_pages_once():
    item = {
        "citations": [
            {"doc_id": "doc1", "page": 5, "section_ref": None, "verified": True},
            {"doc_id": "doc1", "page": 5, "section_ref": None, "verified": True},
        ],
        "contexts": [
            {
                "chunk_id": "c1",
                "doc_id": "doc1",
                "page_start": 4,
                "page_end": 6,
                "section": "s1",
                "text": "covers page 5",
            },
            {
                "chunk_id": "c2",
                "doc_id": "doc1",
                "page_start": 10,
                "page_end": 12,
                "section": "s2",
                "text": "does not cover page 5",
            },
            {
                "chunk_id": "c3",
                "doc_id": "doc2",
                "page_start": 1,
                "page_end": 6,
                "section": "s3",
                "text": "wrong doc",
            },
        ],
    }
    passages = cited_passages(item)
    assert len(passages) == 1
    assert passages[0]["doc_id"] == "doc1"
    assert passages[0]["text"] == "covers page 5"


def test_render_escapes_script_close_tag():
    data = {
        "run_date": "2026-01-01",
        "metrics": {"answer_rate": {"k": 1, "n": 1}},
        "closed_metrics": {"answer_rate": {"k": 1, "n": 1}},
        "titles": {},
        "items": [{"id": "t01", "question": "what about </script> tags?"}],
    }
    html = render(data)
    assert html.count("</script>") == 2
    assert "<\\/script>" in html


def test_build_data_on_real_artifacts():
    data = build_data()
    assert len(data["items"]) == 60
    ids = {item["id"] for item in data["items"]}
    assert ids == {f"t{n:02d}" for n in range(1, 61)}
