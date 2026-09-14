"""Provider-free validation and state primitives for sealed evaluation runs.

This module deliberately stops at preflight, ledger claim, journaling, and budget accounting. The
provider-backed runner remains in :mod:`evals.run_release`; it must receive a validated protocol
and a claimed run before constructing any provider client.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.config import CHROMA_DIR, MANIFEST_PATH, PRICING_PER_MILLION_TOKENS, REPORTS_DIR, ROOT
from src.provenance import index_fingerprint, manifest_digest, pipeline_fingerprint


class ProtocolValidationError(ValueError):
    """Raised before provider construction when a sealed protocol is not trustworthy."""


class LedgerClaimError(RuntimeError):
    """Raised when a sealed dataset pair has already been claimed."""


class BudgetExceeded(RuntimeError):
    """Raised before an operation whose conservative estimate exceeds the remaining cap."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def protocol_digest(protocol: dict) -> str:
    return _sha256_bytes(_canonical_json(protocol))


def seal_digest(protocol: dict) -> str:
    unsigned = copy.deepcopy(protocol)
    unsigned.pop("seal_digest", None)
    lifecycle = unsigned.get("lifecycle")
    if isinstance(lifecycle, dict):
        lifecycle.pop("seal_digest", None)
    return protocol_digest(unsigned)


def _resolve_path(value: str | Path, protocol_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    beside_protocol = protocol_path.parent / path
    return beside_protocol if beside_protocol.exists() else ROOT / path


def _dataset_spec(protocol: dict, name: str) -> dict:
    datasets = protocol.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(datasets.get(name), dict):
        raise ProtocolValidationError(f"protocol datasets.{name} is missing")
    return datasets[name]


def _read_dataset(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        try:
            return [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
        except json.JSONDecodeError as exc:
            raise ProtocolValidationError(f"dataset is not valid JSONL: {path}: {exc}") from exc
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolValidationError(f"dataset is not valid JSON: {path}: {exc}") from exc
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload["items"]
    raise ProtocolValidationError(f"dataset must be a list or an object with items: {path}")


def _manifest_doc_ids(manifest_path: Path) -> set[str]:
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolValidationError(f"source manifest cannot be read: {exc}") from exc
    if not isinstance(entries, list):
        raise ProtocolValidationError("source manifest must be a list")
    ids = {entry.get("id") for entry in entries if isinstance(entry, dict)}
    return {doc_id for doc_id in ids if isinstance(doc_id, str)}


def _validate_dataset(
    name: str, spec: dict, protocol_path: Path, supported_doc_ids: set[str]
) -> tuple[Path, str]:
    path_value = spec.get("path")
    if not path_value:
        raise ProtocolValidationError(f"datasets.{name}.path is missing")
    path = _resolve_path(path_value, protocol_path)
    if not path.exists():
        raise ProtocolValidationError(f"datasets.{name} is missing: {path}")
    actual_hash = sha256_file(path)
    if spec.get("sha256") != actual_hash:
        raise ProtocolValidationError(f"datasets.{name} hash mismatch")
    rows = _read_dataset(path)
    if spec.get("count") != len(rows):
        raise ProtocolValidationError(
            f"datasets.{name} count mismatch: expected {spec.get('count')}, actual {len(rows)}"
        )
    ids = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or any(not isinstance(item_id, str) for item_id in ids):
        raise ProtocolValidationError(f"datasets.{name} contains a row without a string id")
    if len(set(ids)) != len(ids):
        raise ProtocolValidationError(f"datasets.{name} contains duplicate ids")
    if spec.get("ids") is not None and spec["ids"] != ids:
        raise ProtocolValidationError(f"datasets.{name} ids do not match the sealed protocol")
    if name == "golden":
        for row in rows:
            for source in row.get("source", []):
                if source.get("doc_id") not in supported_doc_ids:
                    raise ProtocolValidationError(
                        f"datasets.golden item {row['id']} cites unsupported source "
                        f"{source.get('doc_id')!r}"
                    )
                if not isinstance(source.get("page"), int) or source["page"] < 1:
                    raise ProtocolValidationError(
                        f"datasets.golden item {row['id']} has an invalid source page"
                    )
    return path, actual_hash


@dataclass(frozen=True)
class ValidatedProtocol:
    protocol_path: Path
    protocol_digest: str
    dataset_digest: str
    golden_path: Path
    retrieval_path: Path
    pipeline_fingerprint: str
    index_fingerprint: str
    manifest_sha256: str
    store_identity: str | None
    golden_sha256: str | None = None
    retrieval_sha256: str | None = None


def validate_protocol(
    path: Path,
    *,
    current_pipeline: str | None = None,
    current_index: str | None = None,
    current_manifest: str | None = None,
    store_build_path: Path | None = None,
) -> ValidatedProtocol:
    """Validate the sealed bytes and identities without opening a model, provider, or Chroma."""
    path = Path(path)
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolValidationError(f"protocol cannot be read: {exc}") from exc
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 2:
        raise ProtocolValidationError("protocol schema_version must be 2")
    lifecycle = protocol.get("lifecycle")
    if not isinstance(lifecycle, dict) or lifecycle.get("state") != "sealed":
        raise ProtocolValidationError("protocol lifecycle state is not sealed")
    recorded_seal = protocol.get("seal_digest") or lifecycle.get("seal_digest")
    if not recorded_seal:
        raise ProtocolValidationError("sealed protocol has no stored seal digest")
    if recorded_seal != seal_digest(protocol):
        raise ProtocolValidationError("protocol seal digest mismatch")

    manifest_spec = protocol.get("source_manifest")
    manifest_path = (
        _resolve_path(manifest_spec["path"], path)
        if isinstance(manifest_spec, dict) and manifest_spec.get("path")
        else MANIFEST_PATH
    )
    supported_doc_ids = _manifest_doc_ids(manifest_path)
    if isinstance(manifest_spec, dict) and manifest_spec.get("sha256") != sha256_file(
        manifest_path
    ):
        raise ProtocolValidationError("source manifest hash mismatch")

    golden_path, golden_hash = _validate_dataset(
        "golden", _dataset_spec(protocol, "golden"), path, supported_doc_ids
    )
    retrieval_path, retrieval_hash = _validate_dataset(
        "retrieval", _dataset_spec(protocol, "retrieval"), path, supported_doc_ids
    )
    identities = protocol.get("identities")
    if not isinstance(identities, dict):
        raise ProtocolValidationError("protocol identities are missing")
    if not isinstance(identities.get("model_revisions"), dict) or not all(
        isinstance(value, str) and value for value in identities["model_revisions"].values()
    ):
        raise ProtocolValidationError("protocol must record resolved model revisions")
    if not isinstance(protocol.get("metrics"), list) or not protocol["metrics"]:
        raise ProtocolValidationError("protocol must predeclare metrics")
    if not isinstance(protocol.get("selection_rules"), dict):
        raise ProtocolValidationError("protocol must record selection rules")
    provenance = protocol.get("provenance")
    if (
        not isinstance(provenance, dict)
        or not provenance.get("author")
        or not provenance.get("reviewer")
    ):
        raise ProtocolValidationError("protocol author/reviewer provenance is incomplete")
    expected_pipeline = identities.get("pipeline_fingerprint")
    expected_index = identities.get("index_fingerprint")
    expected_manifest = (
        manifest_spec.get("sha256")
        if isinstance(manifest_spec, dict)
        else identities.get("manifest_sha256")
    )
    if not all(
        isinstance(value, str) and value
        for value in (expected_pipeline, expected_index, expected_manifest)
    ):
        raise ProtocolValidationError(
            "protocol must record pipeline, index, and manifest identities"
        )
    actual_pipeline = (
        current_pipeline if current_pipeline is not None else pipeline_fingerprint(k=5)
    )
    actual_index = current_index if current_index is not None else index_fingerprint()
    actual_manifest = current_manifest if current_manifest is not None else manifest_digest()
    if expected_pipeline != actual_pipeline:
        raise ProtocolValidationError("current pipeline fingerprint does not match sealed protocol")
    if expected_index != actual_index:
        raise ProtocolValidationError("current index fingerprint does not match sealed protocol")
    if expected_manifest != actual_manifest:
        raise ProtocolValidationError("current manifest fingerprint does not match sealed protocol")

    store_spec = identities.get("store")
    store_identity = None
    if isinstance(store_spec, dict):
        store_build_path = store_build_path or CHROMA_DIR / "build.json"
        if not store_build_path.exists():
            raise ProtocolValidationError(f"store build record is missing: {store_build_path}")
        try:
            build_record = json.loads(store_build_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolValidationError(f"store build record is invalid: {exc}") from exc
        store_identity = store_spec.get("build_digest") or build_record.get("index_fingerprint")
        if store_spec.get("index_fingerprint") != build_record.get("index_fingerprint"):
            raise ProtocolValidationError("store index identity does not match sealed protocol")
        if store_spec.get("schema") != build_record.get("schema"):
            raise ProtocolValidationError("store build schema does not match sealed protocol")

    return ValidatedProtocol(
        protocol_path=path,
        protocol_digest=protocol_digest(protocol),
        dataset_digest=_sha256_bytes(f"{golden_hash}:{retrieval_hash}".encode()),
        golden_path=golden_path,
        retrieval_path=retrieval_path,
        pipeline_fingerprint=expected_pipeline,
        index_fingerprint=expected_index,
        manifest_sha256=expected_manifest,
        store_identity=store_identity,
        golden_sha256=golden_hash,
        retrieval_sha256=retrieval_hash,
    )


class SealedRunLedger:
    """SQLite claim ledger; the dataset-pair digest is unique across labels/processes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sealed_runs (
                    dataset_digest TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    protocol_digest TEXT NOT NULL,
                    pipeline_fingerprint TEXT NOT NULL,
                    state TEXT NOT NULL,
                    label TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def claim(self, validated: ValidatedProtocol, label: str) -> str:
        run_id = f"sealed-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC).isoformat()
        try:
            with sqlite3.connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO sealed_runs
                    (dataset_digest, run_id, protocol_digest, pipeline_fingerprint, state, label,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'claimed', ?, ?, ?)
                    """,
                    (
                        validated.dataset_digest,
                        run_id,
                        validated.protocol_digest,
                        validated.pipeline_fingerprint,
                        label,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise LedgerClaimError("sealed dataset pair has already been claimed") from exc
        return run_id

    def get(self, run_id: str) -> dict | None:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM sealed_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_state(self, run_id: str, state: str) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE sealed_runs SET state = ?, updated_at = ? WHERE run_id = ?",
                (state, datetime.now(UTC).isoformat(), run_id),
            )


class RunJournal:
    """Atomically replace a small JSON event log after every intent/result transition."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _append(self, event: dict) -> None:
        events = self._read()
        events.append({"timestamp": datetime.now(UTC).isoformat(), **event})
        fd, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(events, handle, indent=2, sort_keys=True)
            os.replace(temporary, self.path)
        except BaseException:
            os.unlink(temporary)
            raise

    def intent(self, operation: str, item_id: str, estimate_usd: float) -> None:
        self._append(
            {
                "event": "intent",
                "operation": operation,
                "item_id": item_id,
                "estimate_usd": estimate_usd,
            }
        )

    def success(
        self, operation: str, item_id: str, payload: dict, actual_usd: float | None = None
    ) -> None:
        self._append(
            {
                "event": "success",
                "operation": operation,
                "item_id": item_id,
                "payload": payload,
                "actual_usd": actual_usd,
            }
        )

    def failure(self, operation: str, item_id: str, error_type: str) -> None:
        self._append(
            {
                "event": "failure",
                "operation": operation,
                "item_id": item_id,
                "error_type": error_type,
            }
        )

    def successful_payloads(self, operation: str) -> dict[str, dict]:
        return {
            event["item_id"]: event["payload"]
            for event in self._read()
            if event.get("event") == "success" and event.get("operation") == operation
        }

    def ambiguous(self) -> list[dict]:
        events = self._read()
        resolved = {
            (e.get("operation"), e.get("item_id"))
            for e in events
            if e.get("event") in {"success", "failure"}
        }
        return [
            event
            for event in events
            if event.get("event") == "intent"
            and (event.get("operation"), event.get("item_id")) not in resolved
        ]


@dataclass
class CostBudget:
    max_cost_usd: float
    reserved_usd: float = 0.0
    actual_usd: float = 0.0

    def reserve(self, estimate_usd: float) -> None:
        if estimate_usd < 0:
            raise ValueError("cost estimate cannot be negative")
        if self.reserved_usd + estimate_usd > self.max_cost_usd + 1e-12:
            raise BudgetExceeded(
                f"next operation needs ${estimate_usd:.6f}, only "
                f"${self.max_cost_usd - self.reserved_usd:.6f} remains"
            )
        self.reserved_usd += estimate_usd

    def record_actual(self, actual_usd: float) -> None:
        if actual_usd < 0:
            raise ValueError("actual cost cannot be negative")
        self.actual_usd += actual_usd


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate one operation from the checked-in price table, never from a provider response."""
    if model not in PRICING_PER_MILLION_TOKENS:
        raise BudgetExceeded(f"no checked provider price is configured for model {model!r}")
    input_price, output_price = PRICING_PER_MILLION_TOKENS[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def write_run_state(path: Path, *, run_id: str, validated: ValidatedProtocol, state: str) -> None:
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "state": state,
        "protocol_digest": validated.protocol_digest,
        "dataset_digest": validated.dataset_digest,
        "pipeline_fingerprint": validated.pipeline_fingerprint,
        "index_fingerprint": validated.index_fingerprint,
        "manifest_sha256": validated.manifest_sha256,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preflight(
    protocol_path: Path,
    output_dir: Path,
    max_cost_usd: float | None,
    *,
    label: str = "sealed-run",
    resume: str | None = None,
) -> tuple[ValidatedProtocol, SealedRunLedger, str]:
    if max_cost_usd is None or max_cost_usd <= 0:
        raise ProtocolValidationError("sealed execution requires a positive --max-cost-usd cap")
    validated = validate_protocol(Path(protocol_path))
    output_dir = Path(output_dir)
    ledger = SealedRunLedger(output_dir / "sealed_runs.sqlite3")
    if resume:
        record = ledger.get(resume)
        if not record:
            raise ProtocolValidationError(f"cannot resume unknown run: {resume}")
        if record["protocol_digest"] != validated.protocol_digest:
            raise ProtocolValidationError("resume protocol digest does not match the claimed run")
        if record["pipeline_fingerprint"] != validated.pipeline_fingerprint:
            raise ProtocolValidationError("resume pipeline identity does not match the claimed run")
        run_id = resume
    else:
        run_id = ledger.claim(validated, label)
    write_run_state(
        output_dir / f"{run_id}.state.json", run_id=run_id, validated=validated, state="claimed"
    )
    return validated, ledger, run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "runs")
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    try:
        validated, _, run_id = preflight(
            args.protocol, args.output_dir, args.max_cost_usd, resume=args.resume
        )
    except (OSError, ProtocolValidationError, LedgerClaimError) as exc:
        print(f"PREFLIGHT FAILED: {exc}")
        return 1
    print(
        f"PREFLIGHT OK: run_id={run_id} protocol={validated.protocol_digest} "
        f"dataset={validated.dataset_digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
