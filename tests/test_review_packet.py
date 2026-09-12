"""scripts/build_review_packet.py against a hand-built run artifact and a local _FakeCollection —
no vector store, no corpus, no real reports/runs/ file touched."""

import json

from scripts.build_review_packet import build_packet


class _FakeCollection:
    """Minimal stand-in covering both collection.get(where=...) (used by fetch_document_page) and
    collection.get(ids=...) (used for the retrieved-chunk summary)."""

    def __init__(self, ids, documents, metadatas):
        self._ids = ids
        self._documents = documents
        self._metadatas = metadatas

    def get(self, where=None, ids=None, include=None):
        if where is not None:
            keep = [i for i, m in enumerate(self._metadatas) if m["doc_id"] == where["doc_id"]]
        elif ids is not None:
            keep = [self._ids.index(cid) for cid in ids]
        else:
            keep = list(range(len(self._ids)))
        result = {
            "ids": [self._ids[i] for i in keep],
            "documents": [self._documents[i] for i in keep],
            "metadatas": [self._metadatas[i] for i in keep],
        }
        return result


def _write_run(tmp_path, items, run_id="holdout-abc123", label="v1.1.0-test") -> object:
    from evals.run_release import compute_structural_metrics

    path = tmp_path / "run.json"
    payload = {
        "run_id": run_id,
        "split": "holdout",
        "label": label,
        "started_at": "t0",
        "finished_at": "t1",
        "status": "complete",
        "holdout_sha256": "h",
        "pipeline_fingerprint": "fp",
        "manifest_sha256": "mf",
        "structural_metrics": compute_structural_metrics(items),
        "ragas_means": None,
        "ragas_scored_count": 0,
        "ragas_excluded_refusals": [],
        "ragas_failures": [],
        "items": items,
        "error": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_golden(tmp_path, rows) -> object:
    path = tmp_path / "golden_holdout.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_the_packet_includes_the_actual_chunk_text_for_every_cited_page(tmp_path, monkeypatch):
    collection = _FakeCollection(
        ids=["c1"],
        documents=["The National Credit Act establishes the NCR."],
        metadatas=[{"doc_id": "nca_act_34_2005", "page_start": 1, "page_end": 1, "section": ""}],
    )
    monkeypatch.setattr("scripts.build_review_packet.get_collection", lambda: collection)

    items = [
        {
            "id": "gh01",
            "type": "factual",
            "question": "What does the Act establish?",
            "answer": "It establishes the NCR. [nca_act_34_2005, p.1]",
            "refused": False,
            "citation_contract_pass": True,
            "all_citations_verified": True,
            "citations": [{"doc_id": "nca_act_34_2005", "page": 1, "verified": True}],
            "retrieved_chunk_ids": ["c1"],
        }
    ]
    run_path = _write_run(tmp_path, items)
    golden_path = _write_golden(
        tmp_path,
        [
            {
                "id": "gh01",
                "type": "factual",
                "question": "What does the Act establish?",
                "reference_answer": "The National Credit Regulator.",
                "source": [{"doc_id": "nca_act_34_2005", "page": 1}],
            }
        ],
    )

    packet = build_packet(run_path, golden_path)

    assert "The National Credit Act establishes the NCR." in packet
    assert "The National Credit Regulator." in packet


def test_the_packet_says_so_when_no_chunk_covers_a_cited_page(tmp_path, monkeypatch):
    collection = _FakeCollection(ids=[], documents=[], metadatas=[])
    monkeypatch.setattr("scripts.build_review_packet.get_collection", lambda: collection)

    items = [
        {
            "id": "gh02",
            "type": "factual",
            "question": "q?",
            "answer": "a [doc_x, p.99]",
            "refused": False,
            "citation_contract_pass": True,
            "all_citations_verified": False,
            "citations": [{"doc_id": "doc_x", "page": 99, "verified": False}],
            "retrieved_chunk_ids": [],
        }
    ]
    run_path = _write_run(tmp_path, items)
    golden_path = _write_golden(
        tmp_path,
        [
            {
                "id": "gh02",
                "type": "factual",
                "question": "q?",
                "reference_answer": "ref",
                "source": [],
            }
        ],
    )

    packet = build_packet(run_path, golden_path)

    assert "no chunk in the store covers this page" in packet


def test_a_page_range_citation_is_not_printed_twice(tmp_path, monkeypatch):
    # a [doc, p.1-2] citation is stored page-expanded (two Citation rows, page 1 and page 2) — the
    # packet must dedupe on (doc_id, page), not print the same chunk once per expanded page
    collection = _FakeCollection(
        ids=["c1"],
        documents=["spans pages one and two"],
        metadatas=[{"doc_id": "doc_a", "page_start": 1, "page_end": 2, "section": ""}],
    )
    monkeypatch.setattr("scripts.build_review_packet.get_collection", lambda: collection)

    items = [
        {
            "id": "gh03",
            "type": "factual",
            "question": "q?",
            "answer": "a [doc_a, p.1-2]",
            "refused": False,
            "citation_contract_pass": True,
            "all_citations_verified": True,
            "citations": [
                {"doc_id": "doc_a", "page": 1, "verified": True},
                {"doc_id": "doc_a", "page": 2, "verified": True},
            ],
            "retrieved_chunk_ids": ["c1"],
        }
    ]
    run_path = _write_run(tmp_path, items)
    golden_path = _write_golden(
        tmp_path,
        [
            {
                "id": "gh03",
                "type": "factual",
                "question": "q?",
                "reference_answer": "ref",
                "source": [],
            }
        ],
    )

    packet = build_packet(run_path, golden_path)

    assert packet.count("spans pages one and two") == 1


def test_the_packet_marks_source_notices_as_uncaptured_for_a_pre_45a1aad_artifact(
    tmp_path, monkeypatch
):
    collection = _FakeCollection(ids=[], documents=[], metadatas=[])
    monkeypatch.setattr("scripts.build_review_packet.get_collection", lambda: collection)

    items = [
        {
            "id": "gh04",
            "type": "unanswerable",
            "question": "q?",
            "answer": "I don't have a source for that.",
            "refused": True,
            "citation_contract_pass": True,
            "all_citations_verified": True,
            "citations": [],
            "retrieved_chunk_ids": [],
            # no "source_notices" key at all — matches the real rc2 artifact, predating it
        }
    ]
    run_path = _write_run(tmp_path, items)
    golden_path = _write_golden(
        tmp_path,
        [
            {
                "id": "gh04",
                "type": "unanswerable",
                "question": "q?",
                "reference_answer": "no source",
                "source": [],
            }
        ],
    )

    packet = build_packet(run_path, golden_path)

    assert "not captured — this run predates source-notice recording" in packet


def test_the_packet_pairs_each_generated_answer_with_its_reference_answer(tmp_path, monkeypatch):
    collection = _FakeCollection(ids=[], documents=[], metadatas=[])
    monkeypatch.setattr("scripts.build_review_packet.get_collection", lambda: collection)

    items = [
        {
            "id": "gh05",
            "type": "unanswerable",
            "question": "What is the repo rate?",
            "answer": "I don't have a source for that.",
            "refused": True,
            "citation_contract_pass": True,
            "all_citations_verified": True,
            "citations": [],
            "retrieved_chunk_ids": [],
        }
    ]
    run_path = _write_run(tmp_path, items)
    golden_path = _write_golden(
        tmp_path,
        [
            {
                "id": "gh05",
                "type": "unanswerable",
                "question": "What is the repo rate?",
                "reference_answer": "Not in this corpus.",
                "source": [],
            }
        ],
    )

    packet = build_packet(run_path, golden_path)

    assert "What is the repo rate?" in packet
    assert "Not in this corpus." in packet


def test_header_states_the_holdout_is_sealed_and_should_not_be_edited(tmp_path, monkeypatch):
    collection = _FakeCollection(ids=[], documents=[], metadatas=[])
    monkeypatch.setattr("scripts.build_review_packet.get_collection", lambda: collection)
    items = []
    run_path = _write_run(tmp_path, items)
    golden_path = _write_golden(tmp_path, [])

    packet = build_packet(run_path, golden_path)

    assert "sealed" in packet.lower()
    assert "protocol" in packet.lower()
