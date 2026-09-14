"""evals/render_summary.py against hand-built run artifacts on tmp_path — no LLM, no vector store,
no real reports/runs/ file touched. Covers exactly what a re-render must guarantee: it reproduces
the artifact's own numbers byte-for-byte reproducibly, it refuses a schema-drifted or hand-edited
artifact rather than silently rendering it, and it never touches eval_history.csv (that's
promote_to_canonical's job, not this one's)."""

import json

import pytest

from evals import render_summary
from evals.run_release import ReleaseRun, compute_metrics_v2, compute_structural_metrics


def _write_artifact(path, **overrides) -> dict:
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
    payload = {
        "run_id": "holdout-abc123",
        "split": "holdout",
        "label": "l",
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
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_a_run_artifact_round_trips_through_load_run_without_losing_a_field(tmp_path):
    artifact_path = tmp_path / "run.json"
    payload = _write_artifact(artifact_path)

    run = render_summary.load_run(artifact_path)

    assert isinstance(run, ReleaseRun)
    assert run.to_dict() == payload


def test_load_run_names_the_unexpected_key_when_the_artifact_schema_drifts(tmp_path):
    artifact_path = tmp_path / "run.json"
    _write_artifact(artifact_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["a_field_no_release_run_has"] = True
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(render_summary.ArtifactSchemaError, match="a_field_no_release_run_has"):
        render_summary.load_run(artifact_path)


def test_load_run_names_the_missing_key_when_the_artifact_is_incomplete(tmp_path):
    artifact_path = tmp_path / "run.json"
    _write_artifact(artifact_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    del payload["manifest_sha256"]
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(render_summary.ArtifactSchemaError, match="manifest_sha256"):
        render_summary.load_run(artifact_path)


def test_rendering_refuses_when_stored_metrics_disagree_with_the_items(tmp_path):
    artifact_path = tmp_path / "run.json"
    _write_artifact(artifact_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["structural_metrics"]["answerable_answer_rate"] = 0.0  # tampered
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    run = render_summary.load_run(artifact_path)

    with pytest.raises(render_summary.StaleMetricsError):
        render_summary.render(run, tmp_path / "eval_summary.md")


def test_rendering_writes_the_same_table_write_summary_report_would(tmp_path):
    artifact_path = tmp_path / "run.json"
    _write_artifact(artifact_path)
    run = render_summary.load_run(artifact_path)
    summary_path = tmp_path / "eval_summary.md"

    render_summary.render(run, summary_path)

    text = summary_path.read_text(encoding="utf-8")
    assert "holdout-abc123" in text
    assert "95% CI" in text


def test_rendering_never_creates_an_eval_history_csv(tmp_path):
    artifact_path = tmp_path / "run.json"
    _write_artifact(artifact_path)
    run = render_summary.load_run(artifact_path)

    render_summary.render(run, tmp_path / "eval_summary.md")

    assert not (tmp_path / "eval_history.csv").exists()


def test_rerendering_the_sealed_rc2_artifact_reproduces_its_published_numbers():
    # the actual committed release evidence — pins reports/eval_summary.md against accidental
    # drift in either this renderer or the Wilson interval helper it depends on
    from src.config import REPORTS_DIR

    artifact_path = REPORTS_DIR / "runs" / "holdout-0cf5e0821ec7.json"
    run = render_summary.load_run(artifact_path)

    assert run.label == "v1.1.0-rc2"

    published = (REPORTS_DIR / "eval_summary.md").read_text(encoding="utf-8")
    assert "17/24 (71%)" in published
    assert "51%–85%" in published
    assert "6/6 (100%)" in published
    assert "61%–100%" in published
    assert "23/30 (77%)" in published
    assert "30/30 (100%)" in published
    assert "89%–100%" in published


def _write_schema2_artifact(path):
    items = [
        {
            "id": "a1",
            "reference_answerable": True,
            "refused": False,
            "served_answer": "supported",
            "raw_model_output": "supported [doc_a, p.1]",
            "refusal_reason": None,
            "structural_validator_pass": True,
            "citations": [{"verified": True}, {"verified": True}],
        },
        {
            "id": "a2",
            "reference_answerable": True,
            "refused": True,
            "served_answer": None,
            "raw_model_output": "I cannot answer",
            "refusal_reason": "no_context",
            "structural_validator_pass": False,
            "citations": [],
        },
        {
            "id": "u1",
            "reference_answerable": False,
            "refused": False,
            "served_answer": "unsupported",
            "raw_model_output": "unsupported [doc_a, p.1]",
            "refusal_reason": None,
            "structural_validator_pass": True,
            "citations": [{"verified": True}],
        },
    ]
    payload = {
        "schema_version": 2,
        "run_id": "holdout-v2abc",
        "split": "holdout",
        "label": "schema2",
        "started_at": "t0",
        "finished_at": "t1",
        "status": "complete",
        "holdout_sha256": "h",
        "pipeline_fingerprint": "fp",
        "manifest_sha256": "mf",
        "structural_metrics": compute_metrics_v2(items),
        "ragas_means": None,
        "ragas_scored_count": 0,
        "ragas_excluded_refusals": [],
        "ragas_failures": [],
        "items": items,
        "error": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_schema2_load_recomputes_metrics_and_rejects_tampering(tmp_path):
    artifact_path = tmp_path / "schema2.json"
    payload = _write_schema2_artifact(artifact_path)

    run = render_summary.load_run(artifact_path)
    assert run.schema_version == 2
    assert run.structural_metrics == compute_metrics_v2(run.items)

    payload["structural_metrics"]["task_outcome"]["rate"] = 1.0
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    tampered = render_summary.load_run(artifact_path)
    with pytest.raises(render_summary.StaleMetricsError):
        render_summary.render(tampered, tmp_path / "eval_summary.md")


def test_schema2_summary_keeps_citation_occurrences_without_an_item_ci(tmp_path):
    artifact_path = tmp_path / "schema2.json"
    _write_schema2_artifact(artifact_path)
    run = render_summary.load_run(artifact_path)
    summary_path = tmp_path / "eval_summary.md"

    render_summary.render(run, summary_path)

    text = summary_path.read_text(encoding="utf-8")
    assert "Answer coverage | 1/2 (50%)" in text
    assert "Refusal recall | 0/1 (0%)" in text
    assert "Structurally verified citations | 3/3 (100%) | — (citation occurrences) |" in text
    assert "Task outcome | 1/3 (33%)" in text
    assert "Independently reviewed supported-answer accuracy | unavailable" in text
