import json

import pytest

from evals import check_questions
from evals.check_questions import DEV_COUNTS, check

PAGE = "A credit provider must not enter into a reckless credit agreement with any consumer."
QUOTE = "must not enter into a reckless credit agreement with any"


@pytest.fixture(autouse=True)
def fake_env(tmp_path, monkeypatch):
    monkeypatch.setattr(check_questions, "page_text", lambda doc_id, page: PAGE)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "doc_a"}, {"id": "doc_b"}]), encoding="utf-8")
    monkeypatch.setattr(check_questions, "MANIFEST_PATH", manifest)


def _rows(quote=QUOTE, counts=None, unanswerable_evidence=False):
    rows = []
    n = 0
    for kind, k in (counts or DEV_COUNTS).items():
        for _ in range(k):
            n += 1
            row = {
                "id": f"d{n:02d}",
                "type": kind,
                "question": f"What does obligation number {n} require of a provider?",
                "reference_answer": "It is no longer the case." if kind == "false_premise" else "x",
                "evidence": [
                    {"doc_id": "doc_a", "page": 1, "quote": quote},
                    {"doc_id": "doc_b", "page": 1, "quote": quote},
                ],
            }
            if kind == "unanswerable":
                row["evidence"] = (
                    [{"doc_id": "doc_a", "page": 1, "quote": quote}]
                    if unanswerable_evidence
                    else []
                )
                row["why_unanswerable"] = "The corpus contains nothing on this topic at all."
            rows.append(row)
    return rows


def _write(tmp_path, rows):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_valid_file_passes(tmp_path):
    assert check(_write(tmp_path, _rows())) == []


def test_quote_not_on_page_fails(tmp_path):
    problems = check(_write(tmp_path, _rows(quote="a sentence that is nowhere on the page at all")))
    assert any("not found" in p for p in problems)


def test_wrong_counts_fail(tmp_path):
    counts = dict(DEV_COUNTS, single=11)
    assert any("counts" in p for p in check(_write(tmp_path, _rows(counts=counts))))


def test_unanswerable_with_evidence_fails(tmp_path):
    problems = check(_write(tmp_path, _rows(unanswerable_evidence=True)))
    assert any("unanswerable item has evidence" in p for p in problems)
