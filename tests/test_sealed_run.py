import json

import pytest

from evals import run_release
from evals.sealed_run import (
    BudgetExceeded,
    CostBudget,
    LedgerClaimError,
    ProtocolValidationError,
    RunJournal,
    SealedRunLedger,
    seal_digest,
    sha256_file,
    validate_protocol,
    write_run_state,
)
from src.llm import LLMResponse
from src.rag import Citation, RAGResult
from src.retrieve import RetrievedChunk


def _write_protocol(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "f1",
                        "type": "factual",
                        "question": "first",
                        "reference_answer": "ref",
                        "source": [{"doc_id": "doc_a", "page": 1}],
                    }
                ),
                json.dumps(
                    {
                        "id": "u1",
                        "type": "unanswerable",
                        "question": "second",
                        "reference_answer": "none",
                        "source": [],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(json.dumps([{"id": "r1", "question": "first"}]), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "doc_a"}]), encoding="utf-8")
    build = tmp_path / "build.json"
    build.write_text(json.dumps({"schema": 2, "index_fingerprint": "index-a"}), encoding="utf-8")

    protocol = {
        "schema_version": 2,
        "protocol_id": "synthetic-v2",
        "lifecycle": {"state": "sealed", "sealed_at": "2026-09-14T00:00:00Z"},
        "datasets": {
            "golden": {
                "path": str(golden),
                "sha256": sha256_file(golden),
                "count": 2,
                "ids": ["f1", "u1"],
            },
            "retrieval": {
                "path": str(retrieval),
                "sha256": sha256_file(retrieval),
                "count": 1,
                "ids": ["r1"],
            },
        },
        "source_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "identities": {
            "pipeline_fingerprint": "pipeline-a",
            "index_fingerprint": "index-a",
            "manifest_sha256": sha256_file(manifest),
            "model_revisions": {"embedding": "embedding@abc", "answer": "answer@def"},
            "store": {"schema": 2, "index_fingerprint": "index-a"},
        },
        "metrics": ["answer_coverage", "task_outcome"],
        "selection_rules": {"predeclared": True},
        "provenance": {"author": "synthetic-author", "reviewer": "synthetic-reviewer"},
    }
    protocol["lifecycle"]["seal_digest"] = seal_digest(protocol)
    path = tmp_path / "protocol_v2.json"
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return path, protocol, golden, retrieval, manifest, build


def test_validate_protocol_checks_seal_bytes_counts_sources_and_identities(tmp_path):
    path, _, golden, retrieval, manifest, build = _write_protocol(tmp_path)

    validated = validate_protocol(
        path,
        current_pipeline="pipeline-a",
        current_index="index-a",
        current_manifest=sha256_file(manifest),
        store_build_path=build,
    )

    assert validated.golden_path == golden
    assert validated.retrieval_path == retrieval
    assert validated.golden_sha256 == sha256_file(golden)


def test_validate_protocol_rejects_tampered_dataset_and_changed_pipeline(tmp_path):
    path, protocol, golden, _, manifest, build = _write_protocol(tmp_path)
    golden.write_text(golden.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ProtocolValidationError, match="golden.*hash mismatch"):
        validate_protocol(
            path,
            current_pipeline="pipeline-a",
            current_index="index-a",
            current_manifest=sha256_file(manifest),
            store_build_path=build,
        )

    golden.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "id": "f1",
                    "type": "factual",
                    "question": "first",
                    "reference_answer": "ref",
                    "source": [{"doc_id": "doc_a", "page": 1}],
                },
                {
                    "id": "u1",
                    "type": "unanswerable",
                    "question": "second",
                    "reference_answer": "none",
                    "source": [],
                },
            ]
        ),
        encoding="utf-8",
    )
    protocol["identities"]["pipeline_fingerprint"] = "pipeline-b"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ProtocolValidationError, match="seal digest mismatch"):
        validate_protocol(
            path,
            current_pipeline="pipeline-a",
            current_index="index-a",
            current_manifest=sha256_file(manifest),
            store_build_path=build,
        )


def test_ledger_claim_is_unique_even_when_labels_differ(tmp_path):
    path, _, _, _, manifest, build = _write_protocol(tmp_path)
    validated = validate_protocol(
        path,
        current_pipeline="pipeline-a",
        current_index="index-a",
        current_manifest=sha256_file(manifest),
        store_build_path=build,
    )
    ledger_path = tmp_path / "ledger.sqlite3"
    first = SealedRunLedger(ledger_path).claim(validated, "first-label")

    with pytest.raises(LedgerClaimError):
        SealedRunLedger(ledger_path).claim(validated, "second-label")

    assert SealedRunLedger(ledger_path).get(first)["state"] == "claimed"


def test_journal_distinguishes_success_failure_and_ambiguous_intent(tmp_path):
    journal = RunJournal(tmp_path / "run.journal.json")
    journal.intent("generation", "item-1", 0.01)
    assert journal.ambiguous()
    journal.success("generation", "item-1", {"answer": "saved"}, actual_usd=0.002)
    journal.intent("generation", "item-2", 0.01)
    journal.failure("generation", "item-2", "RuntimeError")

    assert journal.successful_payloads("generation") == {"item-1": {"answer": "saved"}}
    assert journal.ambiguous() == []


def test_budget_reserves_before_crossing_the_cap():
    budget = CostBudget(0.10)
    budget.reserve(0.06)
    budget.record_actual(0.04)
    with pytest.raises(BudgetExceeded):
        budget.reserve(0.05)


def test_run_state_contains_only_non_secret_provenance(tmp_path):
    path, _, _, _, manifest, build = _write_protocol(tmp_path)
    validated = validate_protocol(
        path,
        current_pipeline="pipeline-a",
        current_index="index-a",
        current_manifest=sha256_file(manifest),
        store_build_path=build,
    )
    state_path = tmp_path / "state.json"
    write_run_state(state_path, run_id="sealed-1", validated=validated, state="claimed")
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert state["run_id"] == "sealed-1"
    assert "question" not in state
    assert "answer" not in state


def _result(question):
    return RAGResult(
        question=question,
        answer="Answer [doc_a, p.1]",
        citations=[Citation("doc_a", 1, True)],
        retrieved_chunks=[RetrievedChunk("c1", "doc_a", "text", 1, 1, "", 0.9)],
        refused=False,
        flagged_injection=False,
        llm_response=LLMResponse(
            text="Answer [doc_a, p.1]", model="fake", input_tokens=1, output_tokens=1, cost_usd=0
        ),
    )


@pytest.mark.asyncio
async def test_real_generation_exception_keeps_prior_output_and_resume_skips_it(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": item_id,
                    "type": "factual",
                    "question": question,
                    "reference_answer": "ref",
                }
            )
            for item_id, question in (("first", "first"), ("second", "second"))
        ),
        encoding="utf-8",
    )
    journal = RunJournal(tmp_path / "run.journal.json")
    calls = []

    def flaky_answer(question):
        calls.append(question)
        if question == "second":
            raise RuntimeError("synthetic provider failure")
        return _result(question)

    async def judge(question, answer, contexts, reference):
        return {
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
        }

    partial = await run_release.execute_run(
        "dev", "synthetic", flaky_answer, judge, golden_path=golden, journal=journal
    )
    assert partial.status == "partial"
    assert [item["id"] for item in partial.items] == ["first"]
    assert partial.ragas_failures == []
    assert "second" in partial.error

    resumed_calls = []

    def resume_answer(question):
        resumed_calls.append(question)
        return _result(question)

    resumed = await run_release.execute_run(
        "dev",
        "synthetic",
        resume_answer,
        judge,
        golden_path=golden,
        journal=journal,
        resume_items={"first": partial.items[0]},
        resume_judgments=journal.successful_payloads("judge"),
        run_id_override="dev-resumed",
    )

    assert resumed.status == "complete"
    assert resumed_calls == ["second"]


@pytest.mark.asyncio
async def test_insufficient_budget_stops_before_the_next_generation_call(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": "only",
                "type": "factual",
                "question": "only",
                "reference_answer": "ref",
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def answer(question):
        calls.append(question)
        return _result(question)

    async def judge(question, answer, contexts, reference):
        return {
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
        }

    run = await run_release.execute_run(
        "dev",
        "budgeted",
        answer,
        judge,
        golden_path=golden,
        budget=CostBudget(0.01),
        generation_estimate_usd=0.02,
    )

    assert run.status == "partial"
    assert calls == []
    assert "BudgetExceeded" in run.error
