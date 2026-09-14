"""Evidence lifecycle checks for historical, development, and release results.

Historical artifacts are immutable observations.  A development candidate may be rerun while
behaviour changes, but it is not release evidence.  A release candidate must point to its own
complete run and current snapshot, and both references are checked against the bytes and inputs
on disk before any release claim is made.
"""

import hashlib
import json
from pathlib import Path

from evals.snapshot import find_stale_inputs
from src.config import EVALS_DIR, ROOT
from src.provenance import manifest_digest, pipeline_fingerprint

REGISTRY_PATH = EVALS_DIR / "evidence_registry.json"


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    """Load a schema-1 evidence registry without constructing a provider."""
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("evidence registry must be a JSON object")
    if registry.get("schema") != 1:
        raise ValueError(f"unsupported evidence registry schema: {registry.get('schema')!r}")
    return registry


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_errors(label: str, path_value: str | None, expected: str | None) -> list[str]:
    if not path_value:
        return [f"{label} path is missing"]
    path = _resolve(path_value)
    if not path.exists():
        return [f"{label} is missing: {path_value}"]
    if expected and _sha256(path) != expected:
        return [f"{label} hash mismatch"]
    return []


def _historical_errors(registry: dict) -> list[str]:
    errors: list[str] = []
    for entry in registry.get("historical", []):
        if not isinstance(entry, dict):
            errors.append("historical evidence entry is not an object")
            continue
        path_value = entry.get("path")
        if not path_value:
            errors.append("historical artifact path is missing")
            continue
        path = _resolve(path_value)
        if not path.exists():
            errors.append(f"historical artifact is missing: {path_value}")
        elif entry.get("sha256") != _sha256(path):
            errors.append(f"historical artifact hash mismatch: {path_value}")
    return errors


def _candidate_errors(candidate: dict) -> list[str]:
    state = candidate.get("state")
    if state == "development":
        return ["candidate is still in development"]
    if state != "release":
        return [f"candidate state is not release: {state!r}"]

    errors: list[str] = []
    run_path_value = candidate.get("run_path")
    snapshot_path_value = candidate.get("snapshot_path")
    errors.extend(
        _hash_errors("candidate run artifact", run_path_value, candidate.get("run_sha256"))
    )
    errors.extend(
        _hash_errors(
            "candidate snapshot artifact", snapshot_path_value, candidate.get("snapshot_sha256")
        )
    )
    if errors:
        return errors

    run_path = _resolve(run_path_value)
    snapshot_path = _resolve(snapshot_path_value)
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"candidate run artifact cannot be read: {exc}"]
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"candidate snapshot artifact cannot be read: {exc}"]

    if run.get("status") != "complete":
        errors.append("candidate run is not complete")
    if run.get("schema_version") != 2:
        errors.append("candidate run is not a schema-2 release artifact")
    metrics = run.get("structural_metrics")
    if isinstance(metrics, dict):
        for name in (
            "reviewed_supported_answer_accuracy",
            "reviewed_supported_answer_yield",
        ):
            if metrics.get(name) is None:
                errors.append(f"candidate run has no completed review field: {name}")
    else:
        errors.append("candidate run has no structural metrics")

    protocol_path = candidate.get("protocol_path")
    if protocol_path:
        errors.extend(
            _hash_errors("candidate protocol", protocol_path, candidate.get("protocol_sha256"))
        )

    k = candidate.get("k", 5)
    try:
        current_pipeline = pipeline_fingerprint(k=int(k))
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"current pipeline identity cannot be computed: {exc}")
    else:
        if run.get("pipeline_fingerprint") != current_pipeline:
            errors.append("candidate run pipeline fingerprint is stale")
        if candidate.get("pipeline_fingerprint") not in (None, current_pipeline):
            errors.append("candidate registry pipeline fingerprint is stale")

    try:
        current_manifest = manifest_digest()
    except OSError as exc:
        errors.append(f"current manifest identity cannot be computed: {exc}")
    else:
        if run.get("manifest_sha256") != current_manifest:
            errors.append("candidate run manifest fingerprint is stale")

    metadata = snapshot.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("candidate snapshot has no metadata")
    else:
        try:
            stale = find_stale_inputs(metadata)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"candidate snapshot provenance cannot be checked: {exc}")
        else:
            if stale:
                errors.append(f"candidate snapshot is stale: {', '.join(stale)}")

    records = snapshot.get("records")
    if not isinstance(records, list) or not records:
        errors.append("candidate snapshot has no recorded items")
    return errors


def current_evidence_errors(registry: dict) -> list[str]:
    """Return named reasons why the registry cannot support a current release claim."""
    errors: list[str] = []
    if registry.get("schema") != 1:
        errors.append(f"unsupported evidence registry schema: {registry.get('schema')!r}")
    errors.extend(_historical_errors(registry))
    candidate = registry.get("candidate")
    if not isinstance(candidate, dict):
        errors.append("candidate entry is missing")
    else:
        errors.extend(_candidate_errors(candidate))
    return errors
