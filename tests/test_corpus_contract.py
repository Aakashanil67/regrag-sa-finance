"""Schema, metadata, and known-source-error regression tests for corpus/manifest.json.

These guard the two failure modes the audit actually found: a manifest entry with a field the
schema wouldn't catch (a made-up status value, a relative URL, a truncated checksum), and a
manifest entry whose *content* is wrong even though its shape is fine (an anachronistic issuer, a
consultation draft with no stage marker). Schema tests use synthetic entries; regression tests read
the real, committed manifest.
"""

import json

import pytest

from scripts.validate_manifest import ManifestValidationError, load_manifest, validate_manifest
from src.config import MANIFEST_PATH


def _valid_entry(**overrides) -> dict:
    entry = {
        "id": "test_doc",
        "title": "Test Document",
        "filename": "test_doc.pdf",
        "download_url": "https://example.gov.za/test_doc.pdf",
        "landing_page_url": "https://example.gov.za/publications/test_doc",
        "referer": "https://example.gov.za/",
        "sha256": "a" * 64,
        "publisher": "Example Regulator",
        "issuing_authority": "Example Regulator",
        "document_type": "Directive",
        "authority_level": "binding_regulatory_instrument",
        "publication_stage": "final",
        "published_date": "2023-01-01",
        "current_status": "current",
        "status_as_of": "2026-01-01",
        "status_source_url": "https://example.gov.za/publications/test_doc",
        "is_third_party": False,
    }
    entry.update(overrides)
    return entry


@pytest.fixture
def manifest():
    return load_manifest()


def test_a_fully_valid_entry_passes():
    validate_manifest([_valid_entry()])


@pytest.mark.parametrize(
    "field", ["id", "title", "filename", "download_url", "authority_level", "current_status"]
)
def test_missing_required_field_is_rejected(field):
    entry = _valid_entry()
    del entry[field]

    with pytest.raises(ManifestValidationError, match=field):
        validate_manifest([entry])


def test_invalid_authority_level_is_rejected():
    with pytest.raises(ManifestValidationError, match="authority_level"):
        validate_manifest([_valid_entry(authority_level="somewhat_binding")])


def test_invalid_publication_stage_is_rejected():
    with pytest.raises(ManifestValidationError, match="publication_stage"):
        validate_manifest([_valid_entry(publication_stage="proposed")])


def test_invalid_current_status_is_rejected():
    with pytest.raises(ManifestValidationError, match="current_status"):
        validate_manifest([_valid_entry(current_status="probably_fine")])


def test_a_year_only_field_without_published_date_is_rejected():
    entry = _valid_entry()
    entry["year"] = 2023

    with pytest.raises(ManifestValidationError, match="year"):
        validate_manifest([entry])


def test_duplicate_ids_are_rejected():
    with pytest.raises(ManifestValidationError, match="duplicate id"):
        validate_manifest([_valid_entry(), _valid_entry()])


def test_duplicate_filenames_are_rejected():
    with pytest.raises(ManifestValidationError, match="duplicate filename"):
        validate_manifest([_valid_entry(id="doc_a"), _valid_entry(id="doc_b")])


def test_non_https_url_is_rejected():
    with pytest.raises(ManifestValidationError, match="https"):
        validate_manifest([_valid_entry(download_url="http://example.gov.za/test_doc.pdf")])


def test_malformed_sha256_is_rejected():
    with pytest.raises(ManifestValidationError, match="sha256"):
        validate_manifest([_valid_entry(sha256="not-a-real-checksum")])


def test_status_date_before_published_date_is_rejected():
    with pytest.raises(ManifestValidationError, match="status_as_of"):
        validate_manifest([_valid_entry(published_date="2023-06-01", status_as_of="2020-01-01")])


def test_status_source_page_requires_status_source_id():
    with pytest.raises(ManifestValidationError, match="status_source_id"):
        validate_manifest([_valid_entry(status_source_page=3)])


def test_the_committed_manifest_is_schema_valid():
    entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(entries)  # raises on any violation


def test_the_2004_circulars_are_not_attributed_to_the_2018_prudential_authority(manifest):
    for doc_id in (
        "sarb_circular_19_2004_capital_hybrid_instruments",
        "sarb_circular_6_2004_basel_ii_update",
    ):
        entry = manifest[doc_id]
        assert "Prudential Authority" not in entry["issuing_authority"]


def test_nca_act_issuing_authority_is_parliament_not_its_publisher(manifest):
    entry = manifest["nca_act_34_2005"]

    assert "Parliament" in entry["issuing_authority"]
    assert entry["publisher"] == "Department of Trade, Industry and Competition (dtic)"
    assert entry["authority_level"] == "primary_legislation"


def test_the_otc_derivatives_document_is_marked_as_a_consultation_draft(manifest):
    # this is the specific known error the audit found: a "for comments" draft with no stage
    # marker anywhere in the pipeline, letting it read as equivalent to a final standard
    entry = manifest["fsca_conduct_standard_otc_derivatives_2018"]

    assert entry["publication_stage"] == "consultation"
    assert entry["authority_level"] == "consultation_or_discussion"


def test_fsb_era_documents_retain_the_financial_services_board_as_issuer(manifest):
    for doc_id in ("fsca_rdr_2014", "fsca_tcf_2011"):
        entry = manifest[doc_id]
        assert "Financial Services Board" in entry["issuing_authority"]
        assert "Financial Sector Conduct Authority" not in entry["issuing_authority"]


def test_the_conduct_standard_press_release_is_not_labelled_as_the_standard_itself(manifest):
    entry = manifest["fsca_press_conduct_standard_banks_2020"]

    assert entry["document_type"] == "Press Release"
    assert entry["authority_level"] != "binding_regulatory_instrument"


def test_the_pwc_guide_is_flagged_third_party_with_the_status_it_actually_has(manifest):
    entry = manifest["pwc_practical_guide_ifrs9"]

    assert entry["is_third_party"] is True
    assert entry["authority_level"] == "third_party_commentary"


def test_every_status_that_is_not_current_or_unknown_names_evidence(manifest):
    for doc_id, entry in manifest.items():
        if entry["current_status"] in ("withdrawn", "superseded", "historical_snapshot"):
            assert entry[
                "status_source_url"
            ], f"{doc_id} has status {entry['current_status']} with no evidence URL"
