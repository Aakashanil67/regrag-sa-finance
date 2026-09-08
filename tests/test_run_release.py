"""evals/run_release.py against fake answer/judge functions and a temporary reports directory —
no real LLM, vector store, or RAGAS judge. Covers exactly what the release protocol requires: a
generation or judge failure makes the run partial and blocks canonical report promotion; every
rate is computed over its own denominator; the holdout can't run unsealed.
"""

import json

import pytest

from evals import run_release
from src.llm import LLMResponse
from src.rag import Citation, RAGResult, RefusalReason
from src.retrieve import RetrievedChunk


def _result(question, refused=False, citations=None, refusal_reason=None) -> RAGResult:
    citations = citations if citations is not None else [Citation("doc_a", 1, True)]
    return RAGResult(
        question=question,
        answer="An answer with a citation. [doc_a, p.1]" if not refused else run_release.__doc__,
        citations=[] if refused else citations,
        retrieved_chunks=[] if refused else [RetrievedChunk("c1", "doc_a", "text", 1, 1, "", 0.9)],
        refused=refused,
        flagged_injection=False,
        llm_response=LLMResponse(
            text="...", model="fake", input_tokens=1, output_tokens=1, cost_usd=0.0
        ),
        refusal_reason=refusal_reason,
    )


def _golden_items():
    return [
        {"id": "f1", "type": "factual", "question": "f1?", "reference_answer": "ref1"},
        {"id": "f2", "type": "factual", "question": "f2?", "reference_answer": "ref2"},
        {"id": "m1", "type": "multi-doc", "question": "m1?", "reference_answer": "ref3"},
        {"id": "u1", "type": "unanswerable", "question": "u1?", "reference_answer": "no source"},
        {"id": "u2", "type": "unanswerable", "question": "u2?", "reference_answer": "no source"},
    ]


async def _fake_judge_success(question, answer, chunk_ids, reference):
    return {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
        "context_recall": 0.6,
    }


def _write_golden(path, items):
    path.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")


@pytest.fixture
def dev_golden(tmp_path, monkeypatch):
    path = tmp_path / "golden_dev.jsonl"
    _write_golden(path, _golden_items())
    monkeypatch.setitem(run_release._GOLDEN_PATHS, "dev", path)
    return path


def test_compute_structural_metrics_uses_the_right_denominator_for_each_rate():
    items = [
        {
            "type": "factual",
            "refused": False,
            "citation_contract_pass": True,
            "all_citations_verified": True,
        },
        {
            "type": "factual",
            "refused": True,
            "citation_contract_pass": False,
            "all_citations_verified": True,
        },
        {
            "type": "multi-doc",
            "refused": False,
            "citation_contract_pass": True,
            "all_citations_verified": False,
        },
        {
            "type": "unanswerable",
            "refused": True,
            "citation_contract_pass": True,
            "all_citations_verified": True,
        },
        {
            "type": "unanswerable",
            "refused": False,
            "citation_contract_pass": False,
            "all_citations_verified": True,
        },
    ]

    m = run_release.compute_structural_metrics(items)

    assert m["answerable_count"] == 3
    assert m["answerable_answered_count"] == 2
    assert m["answerable_answer_rate"] == pytest.approx(2 / 3)
    assert m["unanswerable_count"] == 2
    assert m["unanswerable_refused_count"] == 1
    assert m["unanswerable_refusal_recall"] == pytest.approx(0.5)
    assert m["citation_contract_pass_count"] == 3
    assert m["citation_contract_pass_rate"] == pytest.approx(3 / 5)
    assert m["verified_citation_count"] == 4
    assert m["verified_citation_rate"] == pytest.approx(4 / 5)


def test_structural_metrics_report_none_not_a_fake_zero_for_an_empty_denominator():
    items = [
        {
            "type": "factual",
            "refused": False,
            "citation_contract_pass": True,
            "all_citations_verified": True,
        }
    ]

    m = run_release.compute_structural_metrics(items)

    assert m["unanswerable_refusal_recall"] is None  # no unanswerable items in this set


def test_run_item_marks_an_answered_answerable_item_as_passing_the_contract():
    row = run_release.run_item(
        {"id": "f1", "type": "factual", "question": "q"}, lambda q: _result(q)
    )

    assert row["citation_contract_pass"] is True
    assert row["all_citations_verified"] is True


def test_run_item_marks_a_correct_refusal_on_an_unanswerable_item_as_passing():
    row = run_release.run_item(
        {"id": "u1", "type": "unanswerable", "question": "q"},
        lambda q: _result(q, refused=True, refusal_reason=RefusalReason.NO_CONTEXT),
    )

    assert row["citation_contract_pass"] is True
    assert row["refused"] is True


def test_run_item_marks_a_refusal_on_an_answerable_item_as_failing_the_contract():
    row = run_release.run_item(
        {"id": "f1", "type": "factual", "question": "q"},
        lambda q: _result(q, refused=True, refusal_reason=RefusalReason.MISSING_CITATION),
    )

    assert row["citation_contract_pass"] is False


@pytest.mark.asyncio
async def test_a_complete_run_has_no_generation_or_judge_failures(dev_golden):
    run = await run_release.execute_run("dev", "test-label", _result, _fake_judge_success)

    assert run.status == "complete"
    assert run.error is None
    assert run.ragas_scored_count == 3  # f1, f2, m1 (factual + multi-doc), not u1/u2
    assert run.ragas_means is not None


@pytest.mark.asyncio
async def test_a_generation_failure_makes_the_run_partial(dev_golden):
    run = await run_release.execute_run(
        "dev", "test-label", _result, _fake_judge_success, item_failures={"f1": "boom"}
    )

    assert run.status == "partial"
    assert "f1" in run.error
    assert len(run.items) == 4  # f2, m1, u1, u2 — f1 excluded, not silently counted as success


@pytest.mark.asyncio
async def test_a_judge_failure_makes_the_run_partial_but_keeps_everything_else(dev_golden):
    async def flaky_judge(question, answer, chunk_ids, reference):
        if question == "f2?":
            raise RuntimeError("judge timed out")
        return await _fake_judge_success(question, answer, chunk_ids, reference)

    run = await run_release.execute_run("dev", "test-label", _result, flaky_judge)

    assert run.status == "partial"
    assert len(run.items) == 5  # generation still succeeded for every item
    assert len(run.ragas_failures) == 1
    assert run.ragas_failures[0]["id"] == "f2"
    assert run.ragas_scored_count == 2  # f1 and m1 scored fine; only f2's judge call failed


@pytest.mark.asyncio
async def test_a_refused_answerable_item_is_excluded_from_ragas_not_scored_as_zero(dev_golden):
    def answer_fn(question):
        if question == "f1?":
            return _result(question, refused=True, refusal_reason=RefusalReason.NO_CONTEXT)
        return _result(question)

    run = await run_release.execute_run("dev", "test-label", answer_fn, _fake_judge_success)

    assert run.ragas_scored_count == 2  # f2 and m1; f1's refusal is excluded, not scored as 0
    assert run.ragas_excluded_refusals == [{"id": "f1", "question": "f1?"}]


def test_write_run_artifact_writes_a_json_file_under_the_runs_dir(tmp_path):
    run = run_release.ReleaseRun(
        run_id="dev-abc123",
        split="dev",
        label="l",
        started_at="t0",
        finished_at="t1",
        status="complete",
        holdout_sha256=None,
        pipeline_fingerprint="fp",
        manifest_sha256="mf",
        structural_metrics={},
        ragas_means=None,
        ragas_scored_count=0,
        ragas_excluded_refusals=[],
        ragas_failures=[],
        items=[],
    )

    path = run_release.write_run_artifact(run, runs_dir=tmp_path / "runs")

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "dev-abc123"


def test_promote_to_canonical_writes_summary_and_appends_history(tmp_path, monkeypatch):
    monkeypatch.setattr(run_release, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(run_release, "EVAL_HISTORY_CSV", tmp_path / "eval_history.csv")
    run = run_release.ReleaseRun(
        run_id="dev-abc123",
        split="dev",
        label="l",
        started_at="t0",
        finished_at="t1",
        status="complete",
        holdout_sha256=None,
        pipeline_fingerprint="fp",
        manifest_sha256="mf",
        structural_metrics=run_release.compute_structural_metrics(
            [
                {
                    "type": "factual",
                    "refused": False,
                    "citation_contract_pass": True,
                    "all_citations_verified": True,
                }
            ]
        ),
        ragas_means=None,
        ragas_scored_count=0,
        ragas_excluded_refusals=[],
        ragas_failures=[],
        items=[
            {
                "type": "factual",
                "refused": False,
                "citation_contract_pass": True,
                "all_citations_verified": True,
            }
        ],
    )

    run_release.promote_to_canonical(run)

    assert (tmp_path / "eval_summary.md").exists()
    history = (tmp_path / "eval_history.csv").read_text(encoding="utf-8")
    assert "dev-abc123" in history


def test_holdout_run_is_refused_when_the_protocol_is_not_sealed(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"sealed": False}), encoding="utf-8")
    monkeypatch.setattr(run_release, "EVAL_PROTOCOL_PATH", protocol_path)

    with pytest.raises(run_release.HoldoutNotSealedError):
        run_release.assert_split_runnable("holdout")


def test_dev_split_never_needs_the_protocol_sealed(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"sealed": False}), encoding="utf-8")
    monkeypatch.setattr(run_release, "EVAL_PROTOCOL_PATH", protocol_path)

    run_release.assert_split_runnable("dev")  # must not raise
