"""evals/run_release.py against fake answer/judge functions and a temporary reports directory —
no real LLM, vector store, or RAGAS judge. Covers exactly what the release protocol requires: a
generation or judge failure makes the run partial and blocks canonical report promotion; every
rate is computed over its own denominator; the holdout can't run unsealed.
"""

import json

import pytest

from evals import run_release
from src.llm import LLMResponse
from src.rag import Citation, RAGResult, RefusalReason, SourceNotice, SourceReference
from src.retrieve import RetrievedChunk


def _result(
    question,
    refused=False,
    citations=None,
    refusal_reason=None,
    source_notices=None,
    llm_text="...",
) -> RAGResult:
    citations = citations if citations is not None else [Citation("doc_a", 1, True)]
    return RAGResult(
        question=question,
        answer="An answer with a citation. [doc_a, p.1]" if not refused else run_release.__doc__,
        citations=[] if refused else citations,
        retrieved_chunks=[] if refused else [RetrievedChunk("c1", "doc_a", "text", 1, 1, "", 0.9)],
        refused=refused,
        flagged_injection=False,
        llm_response=LLMResponse(
            text=llm_text, model="fake", input_tokens=1, output_tokens=1, cost_usd=0.0
        ),
        source_notices=source_notices or [],
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


def test_metrics_v2_separates_verified_citations_from_task_outcome():
    items = [
        {
            "id": "answered-answerable",
            "reference_answerable": True,
            "refused": False,
            "served_answer": "Supported answer",
            "structural_validator_pass": True,
            "citations": [{"verified": True}, {"verified": True}],
            "refusal_reason": None,
        },
        {
            "id": "refused-answerable",
            "reference_answerable": True,
            "refused": True,
            "served_answer": None,
            "structural_validator_pass": False,
            "citations": [],
            "refusal_reason": "no_context",
        },
        {
            "id": "answered-unanswerable",
            "reference_answerable": False,
            "refused": False,
            "served_answer": "Unsupported answer",
            "structural_validator_pass": True,
            "citations": [{"verified": True}],
            "refusal_reason": None,
        },
    ]

    metrics = run_release.compute_metrics_v2(items)

    assert metrics["answer_coverage"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert metrics["refusal_recall"] == {"numerator": 0, "denominator": 1, "rate": 0.0}
    assert metrics["citation_verification"] == {
        "numerator": 3,
        "denominator": 3,
        "rate": 1.0,
    }
    assert metrics["task_outcome"] == {"numerator": 1, "denominator": 3, "rate": 1 / 3}
    assert metrics["structural_pass_rate"] == {"numerator": 2, "denominator": 2, "rate": 1.0}


def test_metrics_v2_uses_null_citation_rates_for_all_refusals_and_no_items():
    all_refused = [
        {
            "reference_answerable": True,
            "refused": True,
            "served_answer": None,
            "structural_validator_pass": False,
            "citations": [],
        },
        {
            "reference_answerable": False,
            "refused": True,
            "served_answer": None,
            "structural_validator_pass": False,
            "citations": [],
        },
    ]

    metrics = run_release.compute_metrics_v2(all_refused)

    assert metrics["citation_verification"]["rate"] is None
    assert metrics["structural_pass_rate"]["rate"] is None
    assert metrics["answer_coverage"] == {"numerator": 0, "denominator": 1, "rate": 0.0}
    assert metrics["refusal_recall"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert metrics["task_outcome"] == {"numerator": 1, "denominator": 2, "rate": 0.5}

    empty = run_release.compute_metrics_v2([])
    assert empty["citation_verification"]["rate"] is None
    assert empty["structural_pass_rate"]["rate"] is None
    assert empty["answer_coverage"]["rate"] is None
    assert empty["refusal_recall"]["rate"] is None
    assert empty["task_outcome"]["rate"] is None


def test_run_item_marks_an_answered_answerable_item_as_passing_the_contract():
    row = run_release.run_item(
        {"id": "f1", "type": "factual", "question": "q"}, lambda q: _result(q)
    )

    assert row["citation_contract_pass"] is True
    assert row["all_citations_verified"] is True
    assert row["served_answer"] is not None
    assert row["reference_answerable"] is True
    assert row["structural_validator_pass"] is True
    assert row["contexts"][0]["text"] == "text"


def test_run_item_marks_a_correct_refusal_on_an_unanswerable_item_as_passing():
    row = run_release.run_item(
        {"id": "u1", "type": "unanswerable", "question": "q"},
        lambda q: _result(q, refused=True, refusal_reason=RefusalReason.NO_CONTEXT),
    )

    assert row["citation_contract_pass"] is True
    assert row["refused"] is True
    assert row["served_answer"] is None
    assert row["reference_answerable"] is False
    assert row["refusal_reason"] == "no_context"


def test_run_item_marks_a_refusal_on_an_answerable_item_as_failing_the_contract():
    row = run_release.run_item(
        {"id": "f1", "type": "factual", "question": "q"},
        lambda q: _result(q, refused=True, refusal_reason=RefusalReason.MISSING_CITATION),
    )

    assert row["citation_contract_pass"] is False


def test_run_item_captures_source_notices_for_manual_audit():
    # the release protocol's manual audit checks "required source notice is present and
    # evidenced" per item — that's only auditable from the run artifact if the artifact actually
    # records the notice, which run_item previously dropped entirely
    notice = SourceNotice(
        kind="withdrawn_source",
        text="doc_a is treated as withdrawn.",
        evidence=[SourceReference(doc_id="doc_b", page=1)],
    )
    row = run_release.run_item(
        {"id": "f1", "type": "factual", "question": "q"},
        lambda q: _result(q, source_notices=[notice]),
    )

    assert row["source_notices"] == [
        {
            "kind": "withdrawn_source",
            "text": "doc_a is treated as withdrawn.",
            "evidence": [{"doc_id": "doc_b", "page": 1}],
        }
    ]


def test_run_item_records_an_empty_list_when_no_source_notice_applies():
    row = run_release.run_item(
        {"id": "f1", "type": "factual", "question": "q"}, lambda q: _result(q)
    )

    assert row["source_notices"] == []


def test_run_item_records_the_raw_pre_validation_model_output_unconditionally():
    # unlike src/obslog.py's opt-in column, an eval artifact already stores the full question and
    # answer for every item with no privacy gate — no flag should be needed here either
    row = run_release.run_item(
        {"id": "f1", "type": "factual", "question": "q"},
        lambda q: _result(
            q, refused=True, llm_text="I don't have a source for that. Here's a hedge though."
        ),
    )

    assert row["raw_model_output"] == "I don't have a source for that. Here's a hedge though."


@pytest.mark.asyncio
async def test_a_complete_run_has_no_generation_or_judge_failures(dev_golden):
    run = await run_release.execute_run("dev", "test-label", _result, _fake_judge_success)

    assert run.status == "complete"
    assert run.schema_version == 2
    assert "task_outcome" in run.structural_metrics
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


@pytest.mark.asyncio
async def test_judge_receives_the_saved_generation_context_not_a_fresh_retrieval(dev_golden):
    captured = []

    def answer_fn(question):
        result = _result(question)
        result.retrieved_chunks = [
            RetrievedChunk("saved", "doc_a", "exact context A", 1, 1, "", 0.9)
        ]
        result.formatted_context = "(doc_a, p.1)\nexact context A"
        return result

    async def judge(question, answer, contexts, reference):
        captured.append(contexts)
        return await _fake_judge_success(question, answer, contexts, reference)

    await run_release.execute_run("dev", "test-label", answer_fn, judge)

    assert captured
    assert all(contexts == ["exact context A"] for contexts in captured)


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


def test_promote_to_canonical_rejects_a_schema1_development_run(tmp_path, monkeypatch):
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

    with pytest.raises(ValueError, match="schema-1"):
        run_release.promote_to_canonical(run)

    assert not (tmp_path / "eval_summary.md").exists()
    assert not (tmp_path / "eval_history.csv").exists()


def test_promote_to_canonical_rejects_partial_schema2_run_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_release, "REPORTS_DIR", tmp_path)
    items = []
    run = run_release.ReleaseRun(
        run_id="holdout-partial",
        split="holdout",
        label="partial",
        started_at="t0",
        finished_at="t1",
        status="partial",
        holdout_sha256="golden-hash",
        pipeline_fingerprint="fp",
        manifest_sha256="mf",
        structural_metrics=run_release.compute_metrics_v2(items),
        ragas_means=None,
        ragas_scored_count=0,
        ragas_excluded_refusals=[],
        ragas_failures=[],
        items=items,
        schema_version=2,
    )

    with pytest.raises(ValueError, match="partial"):
        run_release.promote_to_canonical(run)

    assert not (tmp_path / "eval_summary.md").exists()


def test_schema2_history_appends_without_mutating_legacy_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(run_release, "REPORTS_DIR", tmp_path)
    legacy_path = tmp_path / "eval_history.csv"
    legacy_path.write_text("legacy bytes\n", encoding="utf-8")
    v2_path = tmp_path / "eval_history_v2.csv"
    monkeypatch.setattr(run_release, "EVAL_HISTORY_V2_CSV", v2_path)
    items = [
        {
            "reference_answerable": True,
            "refused": False,
            "served_answer": "answer",
            "structural_validator_pass": True,
            "citations": [{"verified": True}],
        },
        {
            "reference_answerable": False,
            "refused": True,
            "served_answer": None,
            "structural_validator_pass": False,
            "citations": [],
        },
    ]
    run = run_release.ReleaseRun(
        run_id="dev-v2abc",
        split="dev",
        label="schema2",
        started_at="t0",
        finished_at="t1",
        status="complete",
        holdout_sha256=None,
        pipeline_fingerprint="fp",
        manifest_sha256="mf",
        structural_metrics=run_release.compute_metrics_v2(items),
        ragas_means=None,
        ragas_scored_count=0,
        ragas_excluded_refusals=[],
        ragas_failures=[],
        items=items,
        schema_version=2,
    )

    run_release._append_history_v2(run)

    assert legacy_path.read_text(encoding="utf-8") == "legacy bytes\n"
    history = v2_path.read_text(encoding="utf-8")
    assert history.startswith("schema_version,timestamp,run_id,split,label,")
    assert "dev-v2abc" in history


def _minimal_run(**overrides) -> run_release.ReleaseRun:
    items = overrides.pop(
        "items",
        [
            {
                "type": "factual",
                "refused": False,
                "citation_contract_pass": True,
                "all_citations_verified": True,
            }
        ],
    )
    defaults = dict(
        run_id="holdout-abc123",
        split="holdout",
        label="l",
        started_at="t0",
        finished_at="t1",
        status="complete",
        holdout_sha256="h",
        pipeline_fingerprint="fp",
        manifest_sha256="mf",
        structural_metrics=run_release.compute_structural_metrics(items),
        ragas_means=None,
        ragas_scored_count=0,
        ragas_excluded_refusals=[],
        ragas_failures=[],
        items=items,
    )
    defaults.update(overrides)
    return run_release.ReleaseRun(**defaults)


def test_summary_report_renders_a_confidence_interval_beside_every_rate(tmp_path):
    # one unanswerable item alongside the answerable one gives every metric a nonzero
    # denominator, so all four rows get a real Wilson range rather than the empty-denominator dash
    run = _minimal_run(
        items=[
            {
                "type": "factual",
                "refused": False,
                "citation_contract_pass": True,
                "all_citations_verified": True,
            },
            {
                "type": "unanswerable",
                "refused": True,
                "citation_contract_pass": True,
                "all_citations_verified": True,
            },
        ]
    )
    path = tmp_path / "eval_summary.md"

    run_release._write_summary_report(run, path)

    text = path.read_text(encoding="utf-8")
    assert "95% CI" in text
    # every rate here is 1/1 or 2/2 (100%) — Wilson still shows a real range, not (100%, 100%)
    assert text.count("–") == 4


def test_summary_report_shows_a_dash_not_a_crash_for_an_empty_denominator(tmp_path):
    # no unanswerable items in this run at all — unanswerable_refusal_recall's denominator is 0
    run = _minimal_run(
        items=[
            {
                "type": "factual",
                "refused": False,
                "citation_contract_pass": True,
                "all_citations_verified": True,
            }
        ]
    )
    path = tmp_path / "eval_summary.md"

    run_release._write_summary_report(run, path)

    text = path.read_text(encoding="utf-8")
    assert "Unanswerable refusal recall | 0/0 (n/a) | — |" in text


def test_summary_report_orders_ragas_metrics_consistently_regardless_of_dict_order(tmp_path):
    # a run rehydrated from a JSON artifact (written with sort_keys=True) has ragas_means keys in
    # alphabetical order, not _RAGAS_METRIC_NAMES order — the rendered row must not leak that
    run = _minimal_run(
        ragas_means={
            "answer_relevancy": 0.681,
            "context_precision": 0.894,
            "context_recall": 1.0,
            "faithfulness": 0.866,
        },
        ragas_scored_count=17,
        ragas_excluded_refusals=[{}] * 7,
    )
    path = tmp_path / "eval_summary.md"

    run_release._write_summary_report(run, path)

    text = path.read_text(encoding="utf-8")
    assert (
        "faithfulness=0.866, answer_relevancy=0.681, context_precision=0.894, context_recall=1.000"
        in text
    )


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
