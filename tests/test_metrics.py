from evals.metrics import compute, evidence_hit, raw_citations


def _item(item_id, kind, refused, raw="", citations=(), contexts=(), evidence=(), cost=0.01):
    return {
        "id": item_id,
        "type": kind,
        "refused": refused,
        "raw_model_output": raw,
        "refusal_reason": "model_refusal" if refused else None,
        "citations": list(citations),
        "contexts": list(contexts),
        "evidence": list(evidence),
        "usage": {"cost_usd": cost},
    }


CTX = [{"doc_id": "a", "page_start": 3, "page_end": 4}]


def test_raw_citations_counts_pages_outside_the_context():
    assert raw_citations("x [a, p.3] y [a, p.9] z [b, p.1-2]", CTX) == (1, 4)


def test_evidence_hit_any_and_all():
    ev = [{"doc_id": "a", "page": 4}, {"doc_id": "b", "page": 1}]
    assert evidence_hit({"evidence": ev}, CTX) == (True, False)


def test_compute_uses_separate_denominators():
    items = [
        _item(
            "1",
            "single",
            False,
            "[a, p.3]",
            [{"verified": True, "section_ref": "s 1"}],
            CTX,
            [{"doc_id": "a", "page": 3}],
        ),
        _item("2", "multi", True, contexts=CTX, evidence=[{"doc_id": "a", "page": 9}]),
        _item("3", "unanswerable", True),
        _item("4", "unanswerable", False, "[a, p.3]", [{"verified": True}], CTX),
    ]
    m = compute(items)
    assert (m["answer_rate"]["k"], m["answer_rate"]["n"]) == (1, 2)
    assert (m["refusal_recall"]["k"], m["refusal_recall"]["n"]) == (1, 2)
    assert (m["task_outcome"]["k"], m["task_outcome"]["n"]) == (2, 4)
    assert (m["retrieval_any_hit"]["k"], m["retrieval_any_hit"]["n"]) == (1, 2)
    assert m["served_unverified_citations"] == 0
    assert m["cost_usd"] == 0.04
    assert m["answer_rate"]["ci95"][0] < 0.5 < m["answer_rate"]["ci95"][1]
