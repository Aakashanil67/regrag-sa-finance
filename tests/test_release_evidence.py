"""Current evidence readiness is separate from immutable historical observations."""

from evals.evidence import REGISTRY_PATH, current_evidence_errors, load_registry


def test_historical_artifacts_remain_byte_valid():
    registry = load_registry(REGISTRY_PATH)

    errors = current_evidence_errors(registry)

    assert not [error for error in errors if error.startswith("historical artifact")]


def test_registry_does_not_label_reused_holdout_as_current_release():
    registry = load_registry(REGISTRY_PATH)

    assert registry["candidate"]["state"] == "development"
    assert registry["candidate"]["run_path"] is None
    assert registry["candidate"]["snapshot_path"] is None
    assert "candidate is still in development" in current_evidence_errors(registry)
