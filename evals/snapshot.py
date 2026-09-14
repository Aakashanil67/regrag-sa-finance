"""Provenance for evals/fixtures/ci_subset.json — the metadata that lets CI tell "the recorded
snapshot still reflects the tracked code, corpus, and question set" from "someone changed
behaviour-critical inputs without re-running evals/record_fixtures.py".

This is a staleness check, not a live quality gate: a matching fingerprint proves the fixtures
were recorded against the state currently on disk, nothing about whether a hosted model would
still produce the same output today. See evals/test_snapshot_integrity.py's module docstring for
why that distinction matters.
"""

import hashlib
import json

from src.config import GOLDEN_DEV_PATH, ROOT
from src.llm import effective_llm_settings
from src.provenance import index_fingerprint, manifest_digest, pipeline_fingerprint
from src.rag import _SYSTEM_PROMPT, CITATION_CONTRACT_VERSION

# every file whose behaviour the recorded fixture answers actually depend on — a change to any of
# these can change what rag.py/agent.py would generate, so the snapshot must be re-recorded
BEHAVIOR_CRITICAL_FILES = (
    "src/rag.py",
    "src/retrieve.py",
    "src/chunking.py",
    "src/ingest.py",
    "src/config.py",
)

# requirements.txt itself is just `-r requirements-dev.txt` — hashing only the shim would let a
# chromadb, sentence-transformers, anthropic, or ragas version bump in the files it actually
# resolves to change what CI installs and what record_fixtures.py would produce, invisibly to this
# staleness check
REQUIREMENTS_FILES = ("requirements.txt", "requirements-api.txt", "requirements-dev.txt")

CI_SUBSET_IDS = ["g01", "g04", "g12", "g18", "g25", "g36", "g40", "g46", "g50", "g54"]


def _file_sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def _golden_subset_bytes() -> bytes:
    items = {
        json.loads(line)["id"]: line
        for line in GOLDEN_DEV_PATH.read_text(encoding="utf-8").splitlines()
        if line
    }
    selected = [items[item_id] for item_id in CI_SUBSET_IDS]
    return "\n".join(selected).encode("utf-8")


def compute_snapshot_metadata() -> dict:
    settings = effective_llm_settings()
    return {
        "schema": 1,
        "index_fingerprint": index_fingerprint(),
        "pipeline_fingerprint": pipeline_fingerprint(k=5),
        "file_sha256": {path: _file_sha256(path) for path in BEHAVIOR_CRITICAL_FILES},
        "manifest_sha256": manifest_digest(),
        "prompt_sha256": hashlib.sha256(_SYSTEM_PROMPT.encode()).hexdigest(),
        "citation_contract_version": CITATION_CONTRACT_VERSION,
        "provider": settings.provider,
        "model": settings.model,
        "temperature": settings.temperature,
        "golden_subset_sha256": hashlib.sha256(_golden_subset_bytes()).hexdigest(),
        "requirements_sha256": {path: _file_sha256(path) for path in REQUIREMENTS_FILES},
    }


def find_stale_inputs(recorded_metadata: dict) -> list[str]:
    """Returns the names of every input that no longer matches what's on disk — empty means the
    fixture is still trustworthy as a snapshot of current tracked behaviour."""
    current = compute_snapshot_metadata()
    stale = []

    for path, digest in current["file_sha256"].items():
        if recorded_metadata.get("file_sha256", {}).get(path) != digest:
            stale.append(f"file:{path}")
    for path, digest in current["requirements_sha256"].items():
        if recorded_metadata.get("requirements_sha256", {}).get(path) != digest:
            stale.append(f"requirements:{path}")

    for key in (
        "manifest_sha256",
        "index_fingerprint",
        "pipeline_fingerprint",
        "prompt_sha256",
        "citation_contract_version",
        "provider",
        "model",
        "temperature",
        "golden_subset_sha256",
    ):
        if recorded_metadata.get(key) != current[key]:
            stale.append(key)

    return stale
