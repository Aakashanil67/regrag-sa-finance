import csv
import json

import pytest

from evals import run_eval

RECORD = {
    "answer": "a",
    "served_answer": "a",
    "raw_model_output": "a [d, p.1]",
    "refused": False,
    "refusal_reason": None,
    "citations": [{"doc_id": "d", "page": 1, "verified": True, "section_ref": None}],
    "contexts": [
        {
            "chunk_id": "c",
            "doc_id": "d",
            "page_start": 1,
            "page_end": 1,
            "section": None,
            "text": "t",
        }
    ],
    "formatted_context": "t",
    "source_notices": [],
    "usage": {"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.001},
}


def _setup(monkeypatch, tmp_path):
    dev = tmp_path / "dev.jsonl"
    items = [
        {
            "id": str(i),
            "type": "single",
            "question": "q",
            "reference_answer": "r",
            "evidence": [{"doc_id": "d", "page": 1}],
        }
        for i in range(3)
    ]
    dev.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
    monkeypatch.setattr(run_eval, "SPLITS", {"dev": dev, "test": tmp_path / "test.jsonl"})
    monkeypatch.setattr(run_eval, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(run_eval, "HISTORY_CSV", tmp_path / "history.csv")
    monkeypatch.setattr(run_eval, "rag_record", lambda item: dict(RECORD))
    monkeypatch.setattr(run_eval, "estimate_usd", lambda n, kind: 0.01)
    monkeypatch.setattr("src.provenance.pipeline_fingerprint", lambda k: "fp")


def test_run_writes_artifact_report_and_history(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    run_eval.main(["--split", "dev", "--config", "default", "--label", "t", "--max-usd", "1"])
    artifact = json.loads((tmp_path / "runs" / "dev-t.json").read_text(encoding="utf-8"))
    assert len(artifact["items"]) == 3
    assert artifact["metrics"]["answer_rate"]["k"] == 3
    assert (tmp_path / "runs" / "dev-t.md").exists()
    rows = list(csv.DictReader((tmp_path / "history.csv").open(encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["run_id"] == "dev-t"


def test_test_split_needs_a_frozen_pipeline(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    test = tmp_path / "test.jsonl"
    test.write_text("{}", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({"sha256": run_eval._sha256(test), "frozen_pipeline": None}), encoding="utf-8"
    )
    monkeypatch.setattr(run_eval, "PROTOCOL", protocol)
    with pytest.raises(SystemExit):
        run_eval.main(["--split", "test", "--config", "default", "--label", "t", "--max-usd", "1"])


def test_dry_run_writes_nothing(monkeypatch, tmp_path, capsys):
    _setup(monkeypatch, tmp_path)
    run_eval.main(["--split", "dev", "--config", "default", "--label", "t", "--dry-run"])
    assert "dry run" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()
