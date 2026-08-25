"""CI regression gate: checks the recorded fixture responses (evals/fixtures/ci_subset.json)
against the invariants that must hold regardless of prompt tweaks — refusal on unanswerable
questions, at least one citation on answerable ones, and a faithfulness floor.

Runs against recorded fixtures, not a live API call: no ANTHROPIC_API_KEY needed in CI, no cost on
every push, and no flakiness from model non-determinism. The trade-off, made deliberately rather
than glossed over: this gate can't catch a regression the fixtures don't exercise, and it goes
stale if rag.py changes without re-running evals/record_fixtures.py to refresh them. The full
55-item judged run (evals/run_ragas.py) is the real measurement; this is a cheap tripwire between
those full runs, not a replacement for them.
"""

import json

from src.config import EVALS_DIR

FIXTURES_PATH = EVALS_DIR / "fixtures" / "ci_subset.json"
# 0.5, not a rounder-looking 0.7 — picked after seeing the real number, not before. The recorded
# subset's own mean is 0.628 (six non-refused items, one of them a genuine 0.0 outlier that a
# 6-item average can't absorb the way the full 34-item run in reports/eval_summary.md does); 0.5
# leaves room for that outlier and for ordinary run-to-run LLM variance without the gate firing on
# noise, while still catching an actual collapse in citation grounding.
FAITHFULNESS_THRESHOLD = 0.5


def _load_fixtures() -> list[dict]:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


def test_unanswerable_questions_refuse():
    fixtures = _load_fixtures()
    unanswerable = [f for f in fixtures if f["type"] == "unanswerable"]
    assert unanswerable, "fixture file has no unanswerable items to check"

    for record in unanswerable:
        assert (
            record["refused"] is True
        ), f"{record['id']} should have refused: {record['question']!r}"


def test_answerable_questions_have_at_least_one_citation():
    fixtures = _load_fixtures()
    # excludes items that legitimately refused: an answerable-type item can still hit a real
    # retrieval gap and correctly trigger rag.py's own refusal rule (tracked in
    # reports/failure_analysis.md, not here) — that's a different failure mode from "answered
    # with no citation," and conflating the two here would fail this test for the wrong reason.
    answered = [f for f in fixtures if f["type"] != "unanswerable" and not f["refused"]]
    assert answered, "fixture file has no non-refused answerable items to check"

    for record in answered:
        assert (
            len(record["citations"]) >= 1
        ), f"{record['id']} answered with no citations: {record['question']!r}"


def test_verified_citation_rate_is_high():
    fixtures = _load_fixtures()
    citations = [
        c
        for f in fixtures
        if f["type"] != "unanswerable" and not f["refused"]
        for c in f["citations"]
    ]
    assert citations, "no citations recorded across the answered fixtures"

    verified_rate = sum(1 for c in citations if c["verified"]) / len(citations)
    assert (
        verified_rate >= 0.9
    ), f"only {verified_rate:.0%} of citations verified against retrieved pages"


def test_mean_faithfulness_above_threshold():
    fixtures = _load_fixtures()
    scores = [f["faithfulness"] for f in fixtures if "faithfulness" in f]
    assert scores, "no faithfulness scores recorded in the fixture file"

    mean_score = sum(scores) / len(scores)
    assert (
        mean_score >= FAITHFULNESS_THRESHOLD
    ), f"mean faithfulness {mean_score:.3f} fell below the {FAITHFULNESS_THRESHOLD} regression floor"
