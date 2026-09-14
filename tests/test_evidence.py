import hashlib
import json

from evals import evidence


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_release_candidate(tmp_path):
    run_path = tmp_path / "run.json"
    snapshot_path = tmp_path / "snapshot.json"
    run_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "pipeline_fingerprint": "current-pipeline",
                "manifest_sha256": "current-manifest",
            }
        ),
        encoding="utf-8",
    )
    snapshot_path.write_text(
        json.dumps({"metadata": {"schema": 1}, "records": []}), encoding="utf-8"
    )
    return run_path, snapshot_path


def test_development_candidate_is_not_release_ready():
    registry = {
        "schema": 1,
        "candidate": {"state": "development", "run_path": None, "snapshot_path": None},
        "historical": [],
    }

    assert "candidate is still in development" in evidence.current_evidence_errors(registry)


def test_historical_artifact_hash_mismatch_is_reported(tmp_path):
    artifact = tmp_path / "historical.json"
    artifact.write_text("original", encoding="utf-8")
    registry = {
        "schema": 1,
        "candidate": {"state": "development", "run_path": None, "snapshot_path": None},
        "historical": [
            {"path": str(artifact), "sha256": "0" * 64, "kind": "run"},
        ],
    }

    errors = evidence.current_evidence_errors(registry)

    assert any("historical artifact hash mismatch" in error for error in errors)


def test_release_candidate_is_rejected_when_its_run_pipeline_is_stale(tmp_path, monkeypatch):
    run_path, snapshot_path = _write_release_candidate(tmp_path)
    monkeypatch.setattr(evidence, "pipeline_fingerprint", lambda k: "different-pipeline")
    monkeypatch.setattr(evidence, "manifest_digest", lambda: "current-manifest")
    monkeypatch.setattr(evidence, "find_stale_inputs", lambda metadata: [])

    registry = {
        "schema": 1,
        "candidate": {
            "state": "release",
            "run_path": str(run_path),
            "run_sha256": _sha256(run_path),
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": _sha256(snapshot_path),
            "pipeline_fingerprint": "current-pipeline",
            "k": 5,
        },
        "historical": [],
    }

    errors = evidence.current_evidence_errors(registry)

    assert any("candidate run pipeline fingerprint is stale" in error for error in errors)


def test_release_candidate_requires_matching_artifact_hashes(tmp_path, monkeypatch):
    run_path, snapshot_path = _write_release_candidate(tmp_path)
    monkeypatch.setattr(evidence, "pipeline_fingerprint", lambda k: "current-pipeline")
    monkeypatch.setattr(evidence, "manifest_digest", lambda: "current-manifest")
    monkeypatch.setattr(evidence, "find_stale_inputs", lambda metadata: [])

    registry = {
        "schema": 1,
        "candidate": {
            "state": "release",
            "run_path": str(run_path),
            "run_sha256": "0" * 64,
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": "0" * 64,
            "pipeline_fingerprint": "current-pipeline",
            "k": 5,
        },
        "historical": [],
    }

    errors = evidence.current_evidence_errors(registry)

    assert "candidate run artifact hash mismatch" in errors
    assert "candidate snapshot artifact hash mismatch" in errors
