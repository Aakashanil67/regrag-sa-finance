from evals.judge import CORRECTNESS, judge_items, parse_label, summarise


def test_parse_label_variants():
    assert parse_label('{"label": "correct", "reason": "ok"}', CORRECTNESS) == ("correct", "ok")
    assert parse_label('Sure! {"label": "partial", "reason": "half"} done', CORRECTNESS)[0] == "partial"
    assert parse_label("no json here", CORRECTNESS) == ("unparsed", "")
    assert parse_label('{"label": "great"}', CORRECTNESS)[0] == "unparsed"


def _run(closed_book):
    def item(i, kind, refused):
        return {
            "id": i, "type": kind, "refused": refused, "question": "q", "reference_answer": "r",
            "served_answer": None if refused else "a", "formatted_context": "ctx",
        }

    return {
        "closed_book": closed_book,
        "items": [item("1", "single", True), item("2", "single", False), item("3", "unanswerable", False)],
    }


def test_judge_items_call_counts():
    calls = []

    def ask(prompt):
        calls.append(prompt)
        return '{"label": "correct", "reason": "x"}'

    rows = judge_items(_run(False), ask)
    assert len(calls) == 2
    assert [r["correctness"] for r in rows] == ["refused", "correct", "answered_unanswerable"]
    calls.clear()
    judge_items(_run(True), ask)
    assert len(calls) == 1


def test_summarise_counts():
    rows = [
        {"type": "single", "correctness": "correct", "support": "supported"},
        {"type": "single", "correctness": "refused"},
        {"type": "unanswerable", "correctness": "answered_unanswerable"},
    ]
    s = summarise(rows)
    assert s["answerable"] == 2
    assert s["correct"] == 1 and s["refused"] == 1
    assert s["support"]["supported"] == 1
    assert s["answered_unanswerable"] == 1
