"""Structural guarantees for the development/holdout split: no cross-contamination, correct
composition, and — for the holdout specifically — a seal that can't silently drift from the files
it's supposed to freeze. These tests never call a model or retrieve anything; they check the JSON/
JSONL files against each other and against corpus/manifest.json.
"""

import hashlib
import json

import pytest

from src.config import (
    EVAL_PROTOCOL_PATH,
    GOLDEN_DEV_PATH,
    GOLDEN_HOLDOUT_PATH,
    MANIFEST_PATH,
    RETRIEVAL_DEV_PATH,
    RETRIEVAL_HOLDOUT_PATH,
)


def _load_golden(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden_dev():
    return _load_golden(GOLDEN_DEV_PATH)


@pytest.fixture(scope="module")
def golden_holdout():
    return _load_golden(GOLDEN_HOLDOUT_PATH)


@pytest.fixture(scope="module")
def retrieval_dev():
    return _load_json(RETRIEVAL_DEV_PATH)


@pytest.fixture(scope="module")
def retrieval_holdout():
    return _load_json(RETRIEVAL_HOLDOUT_PATH)


@pytest.fixture(scope="module")
def active_manifest_ids():
    return {e["id"] for e in _load_json(MANIFEST_PATH)}


@pytest.fixture(scope="module")
def protocol():
    return _load_json(EVAL_PROTOCOL_PATH)


def test_no_duplicate_question_across_dev_and_holdout(
    golden_dev, golden_holdout, retrieval_dev, retrieval_holdout
):
    dev_questions = {g["question"] for g in golden_dev} | {r["question"] for r in retrieval_dev}
    holdout_questions = {g["question"] for g in golden_holdout} | {
        r["question"] for r in retrieval_holdout
    }
    assert dev_questions & holdout_questions == set()


def test_no_duplicate_id_across_dev_and_holdout(
    golden_dev, golden_holdout, retrieval_dev, retrieval_holdout
):
    dev_ids = {g["id"] for g in golden_dev} | {r["id"] for r in retrieval_dev}
    holdout_ids = {g["id"] for g in golden_holdout} | {r["id"] for r in retrieval_holdout}
    assert dev_ids & holdout_ids == set()


def test_every_active_document_appears_in_retrieval_dev_or_holdout(
    retrieval_dev, retrieval_holdout, active_manifest_ids
):
    covered = {r["doc_id"] for r in retrieval_dev} | {r["doc_id"] for r in retrieval_holdout}
    assert active_manifest_ids <= covered


def test_every_active_document_appears_in_retrieval_holdout_specifically(
    retrieval_holdout, active_manifest_ids
):
    # the holdout must stand on its own as release evidence, not lean on the dev set for coverage
    covered = {r["doc_id"] for r in retrieval_holdout}
    covered |= {r["related_doc_id"] for r in retrieval_holdout if r.get("related_doc_id")}
    assert active_manifest_ids <= covered


def test_answer_holdout_has_the_required_type_composition(golden_holdout):
    assert len(golden_holdout) == 30
    counts = {"factual": 0, "multi-doc": 0, "unanswerable": 0}
    for item in golden_holdout:
        counts[item["type"]] += 1
    assert counts == {"factual": 18, "multi-doc": 6, "unanswerable": 6}


def test_retrieval_holdout_has_thirty_questions_and_enough_cross_document_targets(
    retrieval_holdout,
):
    assert len(retrieval_holdout) == 30
    cross_document = [r for r in retrieval_holdout if r.get("cross_document")]
    assert len(cross_document) >= 6


def test_answerable_holdout_items_reference_an_active_document(golden_holdout, active_manifest_ids):
    for item in golden_holdout:
        if item["type"] == "unanswerable":
            continue
        assert item["source"], f"{item['id']} is answerable but has no source"
        for ref in item["source"]:
            assert ref["doc_id"] in active_manifest_ids, f"{item['id']} cites an inactive document"
            assert isinstance(ref["page"], int) and ref["page"] >= 1


def test_unanswerable_holdout_items_have_no_source_and_a_rationale(golden_holdout):
    for item in golden_holdout:
        if item["type"] != "unanswerable":
            continue
        assert item["source"] == []
        assert item["reference_answer"], f"{item['id']} has no corpus-gap rationale"


def test_answerable_holdout_sources_land_on_a_page_a_real_chunk_covers(golden_holdout):
    from src.chunking import chunk_corpus

    chunks_by_doc = chunk_corpus()
    coverage = {
        doc_id: {p for c in chunks for p in range(c.page_start, c.page_end + 1)}
        for doc_id, chunks in chunks_by_doc.items()
    }
    for item in golden_holdout:
        for ref in item["source"]:
            assert ref["page"] in coverage.get(
                ref["doc_id"], set()
            ), f"{item['id']}: {ref['doc_id']} page {ref['page']} isn't covered by any chunk"


def test_retrieval_holdout_targets_land_on_a_page_a_real_chunk_covers(retrieval_holdout):
    from src.chunking import chunk_corpus

    chunks_by_doc = chunk_corpus()
    coverage = {
        doc_id: {p for c in chunks for p in range(c.page_start, c.page_end + 1)}
        for doc_id, chunks in chunks_by_doc.items()
    }
    for item in retrieval_holdout:
        assert item["page"] in coverage.get(
            item["doc_id"], set()
        ), f"{item['id']}: {item['doc_id']} page {item['page']} isn't covered by any chunk"


def test_protocol_reports_the_actual_split_counts(
    protocol, golden_dev, golden_holdout, retrieval_dev, retrieval_holdout
):
    assert protocol["split_counts"]["golden_dev"] == len(golden_dev)
    assert protocol["split_counts"]["golden_holdout"]["total"] == len(golden_holdout)
    assert protocol["split_counts"]["retrieval_dev"] == len(retrieval_dev)
    assert protocol["split_counts"]["retrieval_holdout"]["total"] == len(retrieval_holdout)


def test_protocol_hash_matches_the_canonical_holdout_bytes(protocol):
    canonical = GOLDEN_HOLDOUT_PATH.read_bytes() + RETRIEVAL_HOLDOUT_PATH.read_bytes()
    assert protocol["holdout_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_sealing_the_protocol_records_the_pipeline_fingerprint(protocol):
    # sealing is a deliberate, separate action, never a side effect of editing the holdout files —
    # an unsealed protocol must carry no fingerprint, and a sealed one must carry a real one
    if protocol["sealed"]:
        assert protocol["pipeline_fingerprint_at_seal"]
        assert len(protocol["pipeline_fingerprint_at_seal"]) == 64
    else:
        assert protocol["pipeline_fingerprint_at_seal"] is None
