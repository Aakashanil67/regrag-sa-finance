"""Validates corpus/manifest.json against the source-governance contract described in
corpus/manifest.schema.json, without adding a jsonschema dependency for one small manifest.

Deliberately stricter than "does it parse": authority_level/publication_stage/current_status are
controlled vocabularies precisely because free-text status fields are how the OTC-derivatives
consultation draft ended up looking citable as a final standard in the first place (see
corpus/README.md and DECISIONS.md) — a typo in a status field should fail loudly here, at fetch/
ingestion/CI time, not silently degrade a citation's authority weight downstream.

    python -m scripts.validate_manifest
"""

import json
import re
import sys
from datetime import date

from src.config import MANIFEST_PATH

_REQUIRED_FIELDS = [
    "id",
    "title",
    "filename",
    "download_url",
    "landing_page_url",
    "referer",
    "sha256",
    "publisher",
    "issuing_authority",
    "document_type",
    "authority_level",
    "publication_stage",
    "published_date",
    "current_status",
    "status_as_of",
    "status_source_url",
    "is_third_party",
]

_AUTHORITY_LEVELS = {
    "primary_legislation",
    "binding_regulatory_instrument",
    "accounting_standard",
    "official_non_binding_guidance",
    "official_explanatory_material",
    "consultation_or_discussion",
    "third_party_commentary",
}

_PUBLICATION_STAGES = {"final", "consultation", "draft"}

_CURRENT_STATUSES = {"current", "withdrawn", "superseded", "historical_snapshot", "unknown"}

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HTTPS_PATTERN = re.compile(r"^https://")
_FILENAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+\.pdf$")


class ManifestValidationError(ValueError):
    """Raised with every error found across the manifest, not just the first — a fetch or CI run
    that stops at the first bad entry hides every other one behind it."""


def _parse_date(value: object, field: str, errors: list[str], doc_id: str) -> date | None:
    if not isinstance(value, str) or not _DATE_PATTERN.match(value):
        errors.append(f"{doc_id}: {field} must be an ISO date (YYYY-MM-DD), got {value!r}")
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        errors.append(f"{doc_id}: {field} is not a valid calendar date: {value!r}")
        return None


def _validate_entry(entry: dict, errors: list[str]) -> None:
    doc_id = entry.get("id", "<missing id>")

    for field in _REQUIRED_FIELDS:
        if field not in entry:
            errors.append(f"{doc_id}: missing required field {field!r}")
    if any(field not in entry for field in _REQUIRED_FIELDS):
        return  # remaining checks assume the fields above exist

    if entry["authority_level"] not in _AUTHORITY_LEVELS:
        errors.append(
            f"{doc_id}: authority_level {entry['authority_level']!r} is not a controlled value"
        )
    if entry["publication_stage"] not in _PUBLICATION_STAGES:
        errors.append(
            f"{doc_id}: publication_stage {entry['publication_stage']!r} is not a controlled value"
        )
    if entry["current_status"] not in _CURRENT_STATUSES:
        errors.append(
            f"{doc_id}: current_status {entry['current_status']!r} is not a controlled value"
        )

    if not _SHA256_PATTERN.match(entry["sha256"]):
        errors.append(f"{doc_id}: sha256 is not a 64-character lowercase hex digest")
    if not _FILENAME_PATTERN.match(entry["filename"]):
        errors.append(
            f"{doc_id}: filename {entry['filename']!r} doesn't look like a local .pdf path"
        )

    for field in ("download_url", "landing_page_url", "referer", "status_source_url"):
        if not _HTTPS_PATTERN.match(entry[field]):
            errors.append(f"{doc_id}: {field} must be an https:// URL, got {entry[field]!r}")

    if not isinstance(entry["is_third_party"], bool):
        errors.append(f"{doc_id}: is_third_party must be a boolean")

    published = _parse_date(entry["published_date"], "published_date", errors, doc_id)
    status_as_of = _parse_date(entry["status_as_of"], "status_as_of", errors, doc_id)
    if published is not None and status_as_of is not None and status_as_of < published:
        errors.append(
            f"{doc_id}: status_as_of ({status_as_of}) is before published_date ({published})"
        )

    if "status_source_page" in entry and "status_source_id" not in entry:
        errors.append(f"{doc_id}: status_source_page given without status_source_id")

    if "year" in entry:
        errors.append(f"{doc_id}: legacy 'year' field present — use published_date instead")


def validate_manifest(entries: list[dict]) -> None:
    errors: list[str] = []

    seen_ids: dict[str, int] = {}
    seen_filenames: dict[str, int] = {}
    for entry in entries:
        _validate_entry(entry, errors)
        doc_id = entry.get("id")
        filename = entry.get("filename")
        if doc_id is not None:
            seen_ids[doc_id] = seen_ids.get(doc_id, 0) + 1
        if filename is not None:
            seen_filenames[filename] = seen_filenames.get(filename, 0) + 1

    for doc_id, count in seen_ids.items():
        if count > 1:
            errors.append(f"duplicate id {doc_id!r} appears {count} times")
    for filename, count in seen_filenames.items():
        if count > 1:
            errors.append(f"duplicate filename {filename!r} appears {count} times")

    if errors:
        raise ManifestValidationError("\n".join(errors))


def load_manifest() -> dict[str, dict]:
    entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {e["id"]: e for e in entries}


def main() -> None:
    entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    try:
        validate_manifest(entries)
    except ManifestValidationError as e:
        for line in str(e).splitlines():
            print(f"ERROR: {line}", file=sys.stderr)
        raise SystemExit(1) from e
    print(f"{len(entries)} manifest entries validated.")


if __name__ == "__main__":
    main()
